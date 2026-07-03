# Distributed Optuna search for teacher v3 (legacy) + make_storage() helper
# (WAL SQLite / pooled Postgres) reused by every later driver.
# Modes: --mode coordinator | worker | report | export-best.

from __future__ import annotations

import os
import sys
import time
import json
import argparse
import warnings
import traceback
from typing import Dict, Optional

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, TeacherV3Params, CHECKPOINT_DIR
from agents.teacher_controller_v3 import (
    TeacherController, evaluate_teacher, save_params, params_from_dict
)
from training.multi_instance_torcs import InstancePool


# ─────────────────────────────────────────────────────────────────────
#  Search space — SINGLE SOURCE OF TRUTH
# ─────────────────────────────────────────────────────────────────────

# Each entry: (low, high, log_scale)
SPEED_SEARCH = {
    # Start/finish straight and approach corners — can be very fast
    "speed_wp0":  (220.0, 300.0, False),  # S/F straight
    "speed_wp1":  (80.0,  180.0, False),  # T1 braking
    "speed_wp2":  (120.0, 240.0, False),  # T2-3 esses
    "speed_wp3":  (60.0,  120.0, False),  # T4 hairpin (slowest)
    "speed_wp4":  (200.0, 290.0, False),  # back straight
    "speed_wp5":  (210.0, 290.0, False),  # back straight
    "speed_wp6":  (130.0, 220.0, False),  # corkscrew approach
    "speed_wp7":  (80.0,  160.0, False),  # corkscrew T8
    "speed_wp8":  (70.0,  200.0, False),  # corkscrew exit T9
    "speed_wp9":  (55.0,  110.0, False),  # Andretti hairpin (slowest)
    "speed_wp10": (70.0,  250.0, False),  # final sector (floor lowered — Turn 11 approach is slow)
    "speed_wp11": (55.0,  200.0, False),  # Turn 11 = SLOW corner (old 200-290 floor forced a crash at ~3275m)
}

TRACKPOS_SEARCH = {
    # Racing line: brake from outside, apex inside, exit outside
    # Positive = right side, negative = left side
    "tp_wp0":  (-0.3, 0.3,  False),  # straight: near center
    "tp_wp1":  ( 0.3, 0.85, False),  # T1 brake: outside (right)
    "tp_wp2":  (-0.8, 0.0,  False),  # T1-2 apex: inside (left)
    "tp_wp3":  ( 0.3, 0.85, False),  # T4 brake: outside (right)
    "tp_wp4":  (-0.8, 0.0,  False),  # T4 apex: inside (left)
    "tp_wp5":  (-0.2, 0.2,  False),  # back straight: center
    "tp_wp6":  ( 0.2, 0.85, False),  # corkscrew brake: outside (right)
    "tp_wp7":  (-0.8, 0.0,  False),  # T8 apex: inside (left)
    "tp_wp8":  (-0.1, 0.7,  False),  # T9: track dependent
    "tp_wp9":  ( 0.3, 0.85, False),  # T10 brake: outside (right)
    "tp_wp10": (-0.8, 0.0,  False),  # T10 apex: inside (left)
    "tp_wp11": (-0.2, 0.4,  False),  # final sector
}

CONTROLLER_SEARCH = {
    # Steering
    "steer_angle_gain":    (15.0, 40.0, False),
    "steer_trackpos_gain": (0.1,  0.8,  False),
    "steer_damping":       (0.1,  0.5,  False),

    # Braking
    "brake_gain":          (1.0,  3.0,  False),
    "brake_lookahead_m":   (40.0, 200.0, False),
    "target_speed_factor": (0.80, 0.98, False),

    # ABS
    "abs_slip_ratio":      (0.10, 0.50, False),
    "abs_brake_cut":       (0.15, 0.55, False),

    # Throttle
    "throttle_in_corner":  (0.30, 0.75, False),
    "accel_on_exit":       (0.70, 1.00, False),
}

# Combined search space
SEARCH_SPACE: Dict[str, tuple] = {
    **SPEED_SEARCH,
    **TRACKPOS_SEARCH,
    **CONTROLLER_SEARCH,
}


def sample_params(trial: optuna.Trial) -> TeacherV3Params:
    # Sample a TeacherV3Params from the Optuna search space.
    kwargs = {}
    for name, (low, high, log) in SEARCH_SPACE.items():
        kwargs[name] = trial.suggest_float(name, low, high, log=log)

    # ABS is always enabled (gives significant advantage)
    kwargs["abs_enabled"] = True

    # Fixed structural params (not tunable — fast enough defaults)
    kwargs["sensor_forward_start"] = 6
    kwargs["sensor_forward_end"]   = 12
    kwargs["sensor_brake_threshold"] = 180.0
    kwargs["sensor_corner_fast"]     = 80.0
    kwargs["sensor_corner_slow"]     = 35.0
    kwargs["launch_throttle"]        = 0.95
    kwargs["launch_steps"]           = 100
    kwargs["throttle_max"]           = 1.0
    kwargs["steer_clip"]             = 1.0
    kwargs["min_brake_speed_kmh"]    = 25.0

    return TeacherV3Params(**kwargs)


# ─────────────────────────────────────────────────────────────────────
#  Single trial
# ─────────────────────────────────────────────────────────────────────

DAGGER_BASELINE_LAP = 106.96   # S3 DAgger v2 best lap (what we must beat)
TARGET_LAP          = 80.0     # Competition target

def run_trial(
    trial:      optuna.Trial,
    pool:       InstancePool,
    n_eval_laps: int   = 3,
    timeout_s:  float  = 420.0,
    verbose:    int    = 1,
) -> float:
    # Evaluate one Optuna trial.
    # Acquires a TORCS slot, runs the teacher for n_eval_laps, returns best lap.
    #
    # Objective: MINIMIZE lap time.
    # Penalty for non-lapping trials: 999 − (max_dist / 36.02) to give
    # Optuna a gradient (partial progress is better than crashing early).
    params = sample_params(trial)

    # Acquire a free TORCS slot (blocks until one is available)
    slot = pool.acquire(timeout=600.0)
    port = slot.port

    try:
        result = evaluate_teacher(
            params    = params,
            port      = port,
            n_laps    = n_eval_laps,
            timeout_s = timeout_s,
            verbose   = verbose > 1,
        )
    except Exception as e:
        print(f"[Trial {trial.number}] Exception on port {port}: {e}")
        result = {"best_lap": float("inf"), "avg_lap": float("inf"),
                  "laps": 0, "max_dist": 0.0}
    finally:
        pool.release(slot)

    best_lap  = result["best_lap"]
    max_dist  = result["max_dist"]
    laps_done = result["laps"]

    if verbose >= 1:
        tag = f"{best_lap:.3f}s" if laps_done > 0 else f"no lap ({max_dist:.0f}m)"
        delta = (f" [{DAGGER_BASELINE_LAP - best_lap:+.2f}s vs DAgger]"
                 if laps_done > 0 else "")
        print(f"[Trial {trial.number:4d}] port={port}  laps={laps_done}  {tag}{delta}")

    # Objective value: lap time (lower = better), or distance-scaled penalty
    if laps_done > 0 and best_lap < float("inf"):
        return best_lap
    else:
        # No lap completed — penalize. Better distance = lower penalty.
        return 999.0 - max_dist / (Config.torcs.track_length_m / 100.0)


# ─────────────────────────────────────────────────────────────────────
#  Storage helpers
# ─────────────────────────────────────────────────────────────────────

def make_storage(storage_url: str):
    # Create Optuna RDB storage. SQLite or PostgreSQL.
    # Normalize Heroku/legacy postgres:// → postgresql:// (SQLAlchemy 1.4+ requires it)
    if storage_url.startswith("postgres://"):
        storage_url = "postgresql://" + storage_url[len("postgres://"):]
    if storage_url.startswith("sqlite:///") and not os.path.isabs(storage_url[10:]):
        # Resolve SQLite path relative to PROJECT_ROOT
        db_file = storage_url[10:]
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), db_file)
        storage_url = f"sqlite:///{db_path}"
    is_sqlite = storage_url.startswith("sqlite")
    # Use 'timeout' for SQLite (busy-timeout), 'connect_timeout' for PostgreSQL
    connect_args = {"timeout": 30} if is_sqlite else {"connect_timeout": 10}

    engine_kwargs = {"pool_pre_ping": True, "connect_args": connect_args}
    if not is_sqlite:
        # Supabase session pooler allows only 15 client connections TOTAL. SQLAlchemy's
        # default pool holds up to ~15 PER process, so a few parallel workers exhaust it.
        # Pin each process to a SINGLE connection so ~12 workers + monitor fit under 15.
        engine_kwargs.update(
            pool_size=1,        # one persistent connection per process
            max_overflow=0,     # never open extra connections
            pool_timeout=30,    # wait up to 30s for the connection instead of erroring
            pool_recycle=280,   # recycle before Supabase drops idle conns (~5 min)
        )

    # skip_compatibility_check: avoids a schema-version WRITE on every load_study.
    # Under many concurrent SQLite writers that write would fail with "database is
    # locked" / "exception during commit". The schema is ours & known-compatible.
    storage = optuna.storages.RDBStorage(
        url=storage_url, engine_kwargs=engine_kwargs, skip_compatibility_check=True)

    if is_sqlite:
        # A LOCAL SQLite DB shared by MANY parallel workers: enable WAL (concurrent
        # readers + one writer) and a long busy-timeout so writers retry instead of
        # crashing with "database is locked". This is what lets ~20 local workers
        # share one file with no network/connection limit.
        from sqlalchemy import event

        @event.listens_for(storage.engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _rec):   # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA busy_timeout=30000;")
            cur.close()

    return storage


def create_study(storage_url: str, study_name: str, n_trials: int = 2000):
    # Create (or load if exists) an Optuna study.
    storage = make_storage(storage_url)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=TPESampler(
            n_startup_trials=40,    # random exploration before TPE kicks in
            multivariate=True,      # model correlations (speed + line are correlated)
            group=True,             # group correlated params
            seed=42,
        ),
        pruner=MedianPruner(
            n_startup_trials=20,
            n_warmup_steps=1,       # prune after 1 intermediate report
            interval_steps=1,
        ),
        load_if_exists=True,
    )
    n_done = len([t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\n[Optuna] Study '{study_name}' ready.")
    print(f"  Completed trials: {n_done} / {n_trials}")
    print(f"  Direction:        minimize (lap time)")
    if n_done > 0:
        try:
            best = study.best_trial
            print(f"  Current best:     {best.value:.3f}s (trial #{best.number})")
        except ValueError:
            pass
    return study


def print_report(storage_url: str, study_name: str):
    # Print a summary of the study's best results.
    study = optuna.load_study(
        study_name=study_name,
        storage=make_storage(storage_url)
    )
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None]

    print(f"\n{'='*60}")
    print(f"  Study: {study_name}")
    print(f"  Completed trials: {len(completed)}")
    print(f"  Total trials:     {len(study.trials)}")

    if not completed:
        print("  No completed trials yet.")
        print(f"{'='*60}\n")
        return

    # Sort by lap time
    completed.sort(key=lambda t: t.value)
    print(f"\n  Top 10 trials:")
    print(f"  {'Rank':4s}  {'Trial':6s}  {'Lap Time':10s}  {'vs Target':10s}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*10}")
    for rank, t in enumerate(completed[:10], 1):
        delta = t.value - TARGET_LAP
        flag  = " ★ SUB-80!" if t.value < TARGET_LAP else ""
        print(f"  {rank:4d}  #{t.number:5d}  {t.value:10.3f}s  "
              f"{delta:+9.3f}s{flag}")

    best = completed[0]
    print(f"\n  Best trial #{best.number}:")
    print(f"    Lap time:  {best.value:.3f}s")
    print(f"    vs Target: {best.value - TARGET_LAP:+.3f}s")
    print(f"    vs DAgger: {best.value - DAGGER_BASELINE_LAP:+.3f}s")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────
#  Export best params
# ─────────────────────────────────────────────────────────────────────

def export_best(storage_url: str, study_name: str, output_path: str):
    # Export the best trial's parameters as JSON for bc_pretrain.py.
    study = optuna.load_study(
        study_name=study_name,
        storage=make_storage(storage_url)
    )
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None and t.value < 900.0]

    if not completed:
        print("[Export] No completed lapping trials to export.")
        return

    best = min(completed, key=lambda t: t.value)
    params = sample_params_from_dict(best.params)
    save_params(params, output_path)

    print(f"\n[Export] Best trial #{best.number}: {best.value:.3f}s")
    print(f"[Export] Params saved → {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Run BC pretraining with v3 teacher:")
    print(f"     python bc_pretrain.py --controller v3 --teacher-params {output_path}")
    print(f"     --n-steps 500000 --output checkpoints/bc_pretrained_v3.pth")
    print(f"  2. Run DAgger:")
    print(f"     python dagger.py --controller v3 --teacher-params {output_path}")
    print(f"     --bc-weights checkpoints/bc_pretrained_v3.pth")
    print(f"     --iterations 5 --steps-per-iter 100000")


def sample_params_from_dict(params_dict: dict) -> TeacherV3Params:
    # Reconstruct TeacherV3Params from Optuna trial.params dict.
    kwargs = dict(params_dict)
    kwargs["abs_enabled"] = True
    kwargs.setdefault("sensor_forward_start", 6)
    kwargs.setdefault("sensor_forward_end", 12)
    kwargs.setdefault("sensor_brake_threshold", 180.0)
    kwargs.setdefault("sensor_corner_fast", 80.0)
    kwargs.setdefault("sensor_corner_slow", 35.0)
    kwargs.setdefault("launch_throttle", 0.95)
    kwargs.setdefault("launch_steps", 100)
    kwargs.setdefault("throttle_max", 1.0)
    kwargs.setdefault("steer_clip", 1.0)
    kwargs.setdefault("min_brake_speed_kmh", 25.0)
    # Filter to valid TeacherV3Params fields
    valid = {f for f in TeacherV3Params.__dataclass_fields__}
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return TeacherV3Params(**filtered)


# ─────────────────────────────────────────────────────────────────────
#  Worker loop
# ─────────────────────────────────────────────────────────────────────

def run_worker(
    storage_url:   str,
    study_name:    str,
    n_instances:   int,
    n_eval_laps:   int   = 3,
    target_trials: int   = 2000,
    verbose:       int   = 1,
):
    # Run Optuna trials indefinitely until the study reaches target_trials.
    #
    # Uses multi-instance parallelism: evaluates up to n_instances trials
    # simultaneously.
    study = optuna.load_study(
        study_name=study_name,
        storage=make_storage(storage_url),
    )

    # Start the TORCS pool
    pool = InstancePool(n_instances=n_instances, verbose=verbose > 0)
    pool.start_all()

    print(f"\n[Worker] Connected to '{study_name}'")
    print(f"[Worker] {n_instances} TORCS instances | target: {target_trials} trials")

    t_start  = time.time()
    count    = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
    executor = ThreadPoolExecutor(max_workers=n_instances)

    try:
        pending: dict = {}  # future → trial

        while True:
            n_done = len([t for t in study.trials
                          if t.state in (optuna.trial.TrialState.COMPLETE,
                                         optuna.trial.TrialState.PRUNED)])
            if n_done >= target_trials:
                print(f"[Worker] Study reached {target_trials} completed trials. Done.")
                break

            # Fill up to n_instances parallel trials
            while len(pending) < n_instances:
                trial = study.ask()
                future = executor.submit(
                    run_trial, trial, pool, n_eval_laps, 420.0, verbose
                )
                pending[future] = trial

            # Wait for any to complete (short timeout so we keep filling)
            done, _ = wait(
                pending.keys(), timeout=5.0,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                trial = pending.pop(future)
                try:
                    lap = future.result()
                    study.tell(trial, lap)
                    count += 1
                    elapsed = time.time() - t_start
                    rate    = count / elapsed * 60
                    print(f"[Worker] Trial {trial.number} done: {lap:.3f}s | "
                          f"total={count} | {rate:.1f} trials/hr")
                except optuna.TrialPruned:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                except Exception as e:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    print(f"[Worker] Trial {trial.number} failed: {e}")

    except KeyboardInterrupt:
        print("\n[Worker] Stopped by user. Completed trials are saved.")

    finally:
        # Cancel remaining futures
        for f in pending:
            f.cancel()
        executor.shutdown(wait=False)
        pool.stop_all()

    elapsed = time.time() - t_start
    print(f"\n[Worker] Ran {count} trials in {elapsed/3600:.1f}h ({count/elapsed*60:.1f} trials/hr)")


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Distributed Optuna HPO for Teacher V3 — Sub-80s Corkscrew"
    )
    ap.add_argument("--mode", required=True,
                    choices=["coordinator", "worker", "report", "export-best"],
                    help=(
                        "coordinator: create study (run once) | "
                        "worker: run trials | "
                        "report: show results | "
                        "export-best: save best params to JSON"
                    ))
    ap.add_argument("--storage",
                    default="sqlite:///optuna_teacher_v3.db",
                    help="SQL storage URL. For distributed use PostgreSQL: "
                         "postgresql://user:pass@host:5432/db")
    ap.add_argument("--study-name", default="teacher_v3",
                    help="Study name (default: teacher_v3)")
    ap.add_argument("--n-trials",   type=int, default=2000,
                    help="Target number of trials (default: 2000)")
    ap.add_argument("--n-instances", type=int, default=None,
                    help="TORCS instances to run in parallel "
                         "(default: Config.multi.n_instances)")
    ap.add_argument("--n-laps",     type=int, default=3,
                    help="Laps per evaluation (default: 3, take best)")
    ap.add_argument("--output",     type=str,
                    default=os.path.join(CHECKPOINT_DIR, "best_teacher_v3.json"),
                    help="Output path for export-best mode")
    ap.add_argument("--verbose",    type=int, default=1,
                    help="Verbosity level (0=quiet, 1=trial summaries, 2=full)")
    args = ap.parse_args()

    if args.mode == "coordinator":
        print(f"\n{'='*60}")
        print(f"  Creating Optuna study: {args.study_name!r}")
        print(f"  Storage: {args.storage}")
        print(f"  Target:  {args.n_trials} trials")
        print(f"{'='*60}")
        create_study(args.storage, args.study_name, args.n_trials)
        print(f"\nStudy created! Next steps:")
        print(f"  1. Run this on EACH machine (worker):")
        print(f"     python optuna_teacher_v3.py --mode worker \\")
        print(f"         --storage \"{args.storage}\" \\")
        print(f"         --study-name \"{args.study_name}\" \\")
        print(f"         --n-instances {Config.multi.n_instances}")
        print(f"  2. Monitor progress:")
        print(f"     python optuna_teacher_v3.py --mode report \\")
        print(f"         --storage \"{args.storage}\" --study-name \"{args.study_name}\"")
        print(f"  3. When done, export best params:")
        print(f"     python optuna_teacher_v3.py --mode export-best \\")
        print(f"         --storage \"{args.storage}\" --study-name \"{args.study_name}\"")

    elif args.mode == "worker":
        n_inst = args.n_instances or Config.multi.n_instances
        print(f"\n{'='*60}")
        print(f"  Starting Optuna Worker")
        print(f"  Study:     {args.study_name}")
        print(f"  Storage:   {args.storage}")
        print(f"  Instances: {n_inst} parallel TORCS processes")
        print(f"  Laps/eval: {args.n_laps}")
        print(f"{'='*60}\n")
        run_worker(
            storage_url   = args.storage,
            study_name    = args.study_name,
            n_instances   = n_inst,
            n_eval_laps   = args.n_laps,
            target_trials = args.n_trials,
            verbose       = args.verbose,
        )

    elif args.mode == "report":
        print_report(args.storage, args.study_name)

    elif args.mode == "export-best":
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        export_best(args.storage, args.study_name, args.output)


if __name__ == "__main__":
    main()

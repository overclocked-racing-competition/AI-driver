# Distributed Optuna tuning for teacher v1/v2 (legacy; superseded by the
# per-version *_linux drivers). Shared SQL study; TPE explore + CMA-ES refine.
# No-lap trials return a 999 s penalty so failures remain informative.
# Modes: --mode create | run | report | refine | export.

import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner
from optuna.storages import RDBStorage, RetryFailedTrialCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
try:
    import agents.teacher_controller as _tc_v1
except ImportError:
    _tc_v1 = None

from core.torcs_env_sac import TorcsSACEnv

# Active controller module — set by --controller in main(). Defaults to v1 so the
# existing study/run behaviour is unchanged unless v2 is explicitly requested.
_CONTROLLER = _tc_v1


def _set_controller(version: str):
    # Select which teacher controller module to tune (v1 / v2 / v3).
    global _CONTROLLER
    if version == "v3":
        import agents.teacher_controller_v3 as m
        _CONTROLLER = m
        print("[Controller] Using teacher_controller_v3 (wider aggression + finish-line fix)")
    elif version == "v2":
        try:
            import agents.teacher_controller_v2 as m
            _CONTROLLER = m
            print("[Controller] Using teacher_controller_v2 (continuous-speed model)")
        except ImportError:
            print("[Controller] WARNING: teacher_controller_v2 not found in this folder. Exporting will work, but running trials will crash.")
            _CONTROLLER = None
    else:
        if _tc_v1 is None:
            print("[Controller] WARNING: teacher_controller (v1) not found in this folder. Exporting will work, but running trials will crash.")
        _CONTROLLER = _tc_v1
        print("[Controller] Using teacher_controller v1")


# ============================================================
# Hyperparameter search space
# ============================================================

def sample_params(trial: optuna.Trial):
    # Dispatch to the active controller's search space (v1 here, or v2's own).
    # v2 ships its own sample_params; v1's lives below as sample_params_v1.
    if hasattr(_CONTROLLER, "sample_params"):
        return _CONTROLLER.sample_params(trial)
    return sample_params_v1(trial)


def sample_params_v1(trial: optuna.Trial):
    # Define the Optuna search space for TeacherParams.
    # All ranges are physically motivated for the Laguna Seca Corkscrew circuit.
    return _tc_v1.TeacherParams(
        # Steering
        steer_gain        = trial.suggest_float("steer_gain",      15.0, 40.0),
        centering_gain    = trial.suggest_float("centering_gain",   0.10, 0.50),
        steer_damp        = trial.suggest_float("steer_damp",       0.0,  0.60),
        racing_line_entry = trial.suggest_float("rl_entry",        -0.60, 0.0),
        racing_line_exit  = trial.suggest_float("rl_exit",          0.0,  0.60),
        entry_sensor_thresh = trial.suggest_float("entry_thr",     40.0, 100.0),
        exit_sensor_thresh  = trial.suggest_float("exit_thr",      80.0, 180.0),

        # Speed management
        straight_thresh  = trial.suggest_float("straight_thr",   120.0, 200.0),
        medium_thresh    = trial.suggest_float("medium_thr",       60.0, 120.0),
        corner_thresh    = trial.suggest_float("corner_thr",       25.0,  70.0),
        straight_accel   = trial.suggest_float("straight_accel",   0.80,  1.0),
        medium_accel     = trial.suggest_float("medium_accel",     0.50,  0.95),
        corner_min_speed = trial.suggest_float("corner_min_spd",   25.0,  55.0),
        corner_max_speed = trial.suggest_float("corner_max_spd",   80.0, 130.0),
        hairpin_min_speed= trial.suggest_float("hairpin_min_spd",  20.0,  45.0),
        hairpin_max_speed= trial.suggest_float("hairpin_max_spd",  40.0,  75.0),
        corner_throttle  = trial.suggest_float("corner_thr_val",   0.10,  0.50),
        hairpin_throttle = trial.suggest_float("hairpin_thr_val",  0.05,  0.30),
        corner_brake_cap = trial.suggest_float("corner_brk",      -1.0,  -0.50),
        hairpin_brake_cap= trial.suggest_float("hairpin_brk",     -1.0,  -0.70),

        # ABS
        abs_enabled         = True,
        abs_slip_threshold  = trial.suggest_float("abs_slip",      1.0,   8.0),
        abs_min_speed       = trial.suggest_float("abs_min_spd",  10.0,  40.0),
        abs_release_fraction= trial.suggest_float("abs_rel",       0.40,  0.90),

        # TCS
        tcs_enabled         = True,
        tcs_slip_threshold  = trial.suggest_float("tcs_slip",      2.0,  12.0),
        tcs_min_speed       = trial.suggest_float("tcs_min_spd",  15.0,  40.0),
        tcs_min_accel_factor= trial.suggest_float("tcs_min_fac",   0.05,  0.30),

        # Launch control
        launch_steps = trial.suggest_int("launch_steps", 5, 30),

        # Gear shift RPMs
        rpm_upshift  = trial.suggest_float("rpm_up",   7000.0, 9000.0),
        rpm_downshift= trial.suggest_float("rpm_down", 2500.0, 4500.0),
    )


# ============================================================
# Single trial evaluation
# ============================================================

CRASH_PENALTY = 999.0   # returned if car doesn't complete a lap
MAX_EVAL_STEPS = 12000  # ~4 minutes; enough to complete a lap at any speed


def evaluate_teacher(params, n_laps: int = 1, verbose: bool = False) -> float:
    # Run the teacher controller in TorcsSACEnv and return the best lap time.
    # Returns CRASH_PENALTY if no lap is completed within MAX_EVAL_STEPS.
    #
    # Parameters
    # ----------
    # params : TeacherParams
    # Controller parameters to evaluate.
    # n_laps : int
    # How many lap times to collect. Only the first (fastest) is returned.
    # verbose : bool
    # Print step-by-step progress.
    #
    # Returns
    # -------
    # float
    # Best lap time in seconds, or CRASH_PENALTY if no lap.
    env = TorcsSACEnv(stage=1)
    teacher = _CONTROLLER.TeacherController(params)

    lap_times = []
    max_dist   = 0.0

    try:
        obs, info = env.reset()
        raw_obs = info.get("raw_obs", {})
        teacher.reset()

        for step in range(MAX_EVAL_STEPS):
            action = teacher.act(raw_obs)
            obs, reward, terminated, truncated, info = env.step(action)
            raw_obs = info.get("raw_obs", {})

            dist = float(raw_obs.get("distRaced", 0.0))
            max_dist = max(max_dist, dist)

            if info.get("lap_completed", False):
                llt = float(info.get("lastLapTime", 0.0))
                if llt > 0:
                    lap_times.append(llt)
                    if verbose:
                        print(f"    LAP {len(lap_times)}: {llt:.2f}s")
                    if len(lap_times) >= n_laps:
                        break

            if terminated or truncated:
                break

    finally:
        env.close()

    if lap_times:
        return float(min(lap_times))

    # No lap: return a penalty that encodes how far the car got
    # (shorter distance = worse penalty), so Optuna can discriminate between crashes.
    fraction = min(max_dist / 3602.0, 0.99)
    return CRASH_PENALTY - fraction * 100.0   # range [899, 999]


def run_trial(trial: optuna.Trial) -> float:
    # Optuna objective: sample params, evaluate, return lap time.
    params = sample_params(trial)
    result = evaluate_teacher(params, n_laps=1)
    return result


# ============================================================
# Storage with heartbeat (self-healing distributed runs)
# ============================================================

# Cache one storage object per process so all functions share a single
# connection pool (important for Supabase's connection limits).
_STORAGE_CACHE = {}


def _make_storage(storage):
    # Wrap a storage URL string in an RDBStorage with heartbeat + auto-retry.
    #
    # Heartbeat makes a multi-machine run self-healing: while a worker runs a trial,
    # a background thread refreshes that trial's heartbeat every `heartbeat_interval`
    # seconds. If the worker dies (crash, power loss, hard kill), its heartbeat goes
    # stale; after `grace_period` seconds another worker marks the trial failed and
    # RetryFailedTrialCallback re-enqueues the same parameters (up to max_retry times).
    # Nothing is silently lost or left stuck in RUNNING state.
    #
    # pool_pre_ping + a small pool keep the Supabase pooler connection healthy and
    # within connection limits when many workers connect at once.
    #
    # A non-string (already an RDBStorage) is returned unchanged. Results are cached
    # per URL so repeated calls in one process reuse the same pool.
    if not isinstance(storage, str):
        return storage
    if storage in _STORAGE_CACHE:
        return _STORAGE_CACHE[storage]

    rdb = RDBStorage(
        url=storage,
        heartbeat_interval=60,     # refresh running trial's heartbeat every 60s
        grace_period=180,          # a trial silent for >180s is considered dead
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
        engine_kwargs={
            "pool_size": 2,
            "max_overflow": 3,
            "pool_pre_ping": True,  # revive dropped cloud-Postgres connections
        },
    )
    _STORAGE_CACHE[storage] = rdb
    return rdb


# ============================================================
# Study management
# ============================================================

def create_study(storage, study_name: str, n_trials: int):
    # Create (or load) a TPE Optuna study for teacher tuning.
    study = optuna.create_study(
        storage=_make_storage(storage),
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(
            n_startup_trials=30,
            multivariate=True,
            group=True,
        ),
        pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=0),
        load_if_exists=True,
    )
    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\n{'='*60}")
    print(f"  Optuna Teacher Study: {study_name}")
    print(f"  Storage: {storage}")
    print(f"  Trials:  {n_done} complete / {n_trials} target")
    print(f"  Objective: minimize lap time (s). Crash → ~999 s")
    print(f"{'='*60}")
    return study


def run_worker(storage, study_name: str, n_trials: int, verbose: bool = False):
    # Connect to the study and run trials until n_trials is reached.
    study = optuna.load_study(storage=_make_storage(storage), study_name=study_name)

    print(f"\n[Worker] Connected to study '{study_name}'")
    print(f"[Worker] Running up to {n_trials} trials...")
    print()

    def _objective(trial):
        t0 = time.time()
        val = run_trial(trial)
        elapsed = time.time() - t0
        lap_str = f"{val:.2f}s" if val < CRASH_PENALTY - 50 else "CRASH"
        print(f"  Trial {trial.number:4d}: {lap_str}  ({elapsed:.0f}s)")
        return val

    try:
        study.optimize(_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    except KeyboardInterrupt:
        # Clean stop on Ctrl+C. The in-progress trial is marked failed by Optuna and
        # retried later by the heartbeat; every COMPLETED trial is already safe in the DB.
        print("\n[Worker] Stopped by user (Ctrl+C). All completed trials are saved. "
              "Safe to close — re-run the same command anytime to continue.")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    laps = [t for t in completed if t.value < CRASH_PENALTY - 50]
    print(f"\n[Worker] Done. {len(completed)} completed, {len(laps)} with a lap time.")
    if laps:
        best = min(laps, key=lambda t: t.value)
        print(f"[Worker] Best lap: {best.value:.2f}s (trial #{best.number})")


def refine_with_cmaes(
    base_storage: str,
    base_study_name: str,
    new_storage: str,
    new_study_name: str,
    n_trials: int = 100,
    sigma0_fraction: float = 0.10,
):
    # Phase 2: CMA-ES local refinement around the best TPE trial (±10%).
    # Creates a new study with CmaEsSampler initialized at the best known params.
    base_study = optuna.load_study(storage=_make_storage(base_storage), study_name=base_study_name)
    best = base_study.best_trial
    print(f"\n[CMA-ES] Starting refinement from trial #{best.number}: {best.value:.2f}s")
    print(f"[CMA-ES] Initial point: {best.params}")

    # Build x0 dict and sigma dict (sigma = fraction of the sampled range)
    x0 = best.params.copy()
    sigma0 = {k: abs(v) * sigma0_fraction for k, v in x0.items() if isinstance(v, float)}

    cmaes_sampler = CmaEsSampler(
        x0=x0,
        sigma0=max(sigma0.values()) if sigma0 else 1.0,
        seed=42,
        n_startup_trials=5,
    )

    study = optuna.create_study(
        storage=_make_storage(new_storage),
        study_name=new_study_name,
        direction="minimize",
        sampler=cmaes_sampler,
        load_if_exists=True,
    )

    def _objective(trial):
        params = sample_params(trial)
        val = evaluate_teacher(params, n_laps=1)
        lap_str = f"{val:.2f}s" if val < CRASH_PENALTY - 50 else "CRASH"
        print(f"  CMA-ES trial {trial.number:3d}: {lap_str}")
        return val

    study.optimize(_objective, n_trials=n_trials, n_jobs=1)

    laps = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
            and t.value < CRASH_PENALTY - 50]
    if laps:
        best_cmaes = min(laps, key=lambda t: t.value)
        print(f"\n[CMA-ES] Best lap after refinement: {best_cmaes.value:.2f}s (trial #{best_cmaes.number})")
        print(f"[CMA-ES] Params: {best_cmaes.params}")
    else:
        print("[CMA-ES] No lap completed in refinement phase.")


def print_report(storage, study_name: str):
    # Print the current best trials and save a JSON report.
    study = optuna.load_study(storage=_make_storage(storage), study_name=study_name)

    all_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    lap_trials = [t for t in all_trials if t.value < CRASH_PENALTY - 50]
    crash_trials = [t for t in all_trials if t.value >= CRASH_PENALTY - 50]

    print(f"\n{'='*60}")
    print(f"  Teacher Optuna Report: {study_name}")
    print(f"  {len(all_trials)} complete, {len(lap_trials)} with a lap, {len(crash_trials)} crashed")
    print(f"{'='*60}")

    if not lap_trials:
        print("  No trials completed a lap yet. Run more trials.")
        return

    lap_trials.sort(key=lambda t: t.value)
    best = lap_trials[0]
    print(f"\n  Best trial (#{best.number}): {best.value:.2f}s")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    print(f"\n  Top 10 lap times:")
    for t in lap_trials[:10]:
        print(f"    #{t.number:4d}: {t.value:.2f}s")

    # Save JSON
    report = {
        "study_name": study_name,
        "n_complete": len(all_trials),
        "n_laps": len(lap_trials),
        "n_crashes": len(crash_trials),
        "best": {"number": best.number, "lap_time": best.value, "params": best.params},
        "top_10": [{"number": t.number, "lap_time": t.value} for t in lap_trials[:10]],
        "generated": datetime.now().isoformat(),
    }
    out = os.path.join(Config.CHECKPOINT_DIR, f"teacher_report_{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {out}")


def export_best_params(storage, study_name: str, out_path: str):
    # Export best trial params as a JSON file for use in bc_pretrain / dagger.
    study = optuna.load_study(storage=_make_storage(storage), study_name=study_name)
    lap_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and t.value < CRASH_PENALTY - 50
    ]
    if not lap_trials:
        print("No completed lap trials to export.")
        return

    best = min(lap_trials, key=lambda t: t.value)
    params_dict = dict(best.params)
    params_dict["_lap_time"] = best.value
    params_dict["_trial_number"] = best.number
    params_dict["_study_name"] = study_name

    with open(out_path, "w") as f:
        json.dump(params_dict, f, indent=2)
    print(f"Best params ({best.value:.2f}s) saved → {out_path}")


def load_params_from_json(path: str):
    # Load TeacherParams from a JSON file exported by export_best_params.
    #
    # Uses the active controller's TeacherParams class. Set the controller with
    # _set_controller("v2") first if the JSON came from a v2 study.
    with open(path) as f:
        d = json.load(f)
    # Strip metadata keys (_lap_time, _trial_number, _study_name)
    d = {k: v for k, v in d.items() if not k.startswith("_")}
    # Optuna stores short param names; map them to dataclass fields.
    if hasattr(_CONTROLLER, "params_from_optuna"):
        return _CONTROLLER.params_from_optuna(d)
    return _CONTROLLER.TeacherParams(**d)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Optuna teacher controller tuning")
    parser.add_argument("--mode", required=True,
                        choices=["create", "run", "report", "refine", "export"],
                        help="create: init study | run: execute trials | "
                             "report: print results | refine: CMA-ES local search | "
                             "export: save best params to JSON")
    parser.add_argument("--storage", default="sqlite:///teacher_study.db",
                        help="SQL storage URL (default: sqlite:///teacher_study.db)")
    parser.add_argument("--study-name", default="teacher_v1")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--base-study", default=None,
                        help="Base study name for CMA-ES refinement (--mode refine)")
    parser.add_argument("--base-storage", default=None,
                        help="Base study storage URL for CMA-ES refinement")
    parser.add_argument("--export-path", default="checkpoints/best_teacher_params.json",
                        help="Output path for --mode export")
    parser.add_argument("--controller", default="v1", choices=["v1", "v2", "v3"],
                        help="Which teacher controller to tune: v1 (discrete zones, default), "
                             "v2 (continuous-speed model), or v3 (wider aggression + finish-line "
                             "fix). Pair with a matching --study-name, e.g. teacher_v3.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Select the controller version (v1 default keeps existing behaviour identical).
    _set_controller(args.controller)

    # Teacher tuning ALWAYS runs with the steering rate-limiter OFF, regardless of the
    # config default. The limiter is a smoothing aid for the imitation/RL pipeline, not
    # a teacher parameter — keeping it off here ensures every trial (old and new, across
    # all machines) measures the same thing, so existing studies stay consistent.
    Config.action.steer_rate_limit_enabled = False

    if args.mode == "create":
        create_study(args.storage, args.study_name, args.n_trials)
        print(f"\nStudy created. Run workers:")
        print(f"  python tune_teacher.py --mode run --storage \"{args.storage}\" "
              f"--study-name \"{args.study_name}\" --n-trials {args.n_trials}")

    elif args.mode == "run":
        create_study(args.storage, args.study_name, args.n_trials)  # no-op if exists
        run_worker(args.storage, args.study_name, args.n_trials, args.verbose)

    elif args.mode == "report":
        print_report(args.storage, args.study_name)

    elif args.mode == "refine":
        if not args.base_study:
            parser.error("--base-study required for --mode refine")
        base_storage = args.base_storage or args.storage
        refine_with_cmaes(
            base_storage=base_storage,
            base_study_name=args.base_study,
            new_storage=args.storage,
            new_study_name=args.study_name,
            n_trials=args.n_trials,
        )

    elif args.mode == "export":
        export_best_params(args.storage, args.study_name, args.export_path)


if __name__ == "__main__":
    main()

# Distributed Optuna HPO for from-scratch/fine-tuned SAC (legacy).
# Shared SQL study; each trial trains 500k steps, evaluates 3 episodes,
# reports best lap; MedianPruner cuts no-lap trials at 200k steps.
# Modes: --mode coordinator | worker | report.

import os
import sys
import time
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage, RetryFailedTrialCallback


# Shared heartbeat storage (self-healing distributed runs across machines).
_STORAGE_CACHE = {}


def _make_storage(storage):
    # Wrap a storage URL in an RDBStorage with heartbeat + auto-retry, so a SAC HPO run
    # distributed across machines is self-healing: if a worker crashes mid-trial, the
    # stale trial is detected after grace_period and its params are retried by another
    # worker. pool_pre_ping + a small pool keep cloud-Postgres connections healthy.
    if not isinstance(storage, str):
        return storage
    if storage in _STORAGE_CACHE:
        return _STORAGE_CACHE[storage]
    rdb = RDBStorage(
        url=storage,
        heartbeat_interval=120,    # SAC trials are long; check in every 2 min
        grace_period=600,          # a trial silent >10 min is considered dead
        failed_trial_callback=RetryFailedTrialCallback(max_retry=2),
        engine_kwargs={"pool_size": 2, "max_overflow": 3, "pool_pre_ping": True},
    )
    _STORAGE_CACHE[storage] = rdb
    return rdb

warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import Config
from core.torcs_env_sac import TorcsSACEnv
from core.custom_policy import LayerNormSACPolicy
from agents.bc_anchored_sac import BCAnchoredSAC
from core.callbacks import FreezeActorCallback

# ==================================================================
# Override Config with trial hyperparameters
# ==================================================================

class TempConfig:
    # Context manager that temporarily overrides Config values.

    def __init__(self, overrides: dict):
        self.overrides = overrides
        self.saved = {}

    def __enter__(self):
        for attr_path, value in self.overrides.items():
            obj = Config
            parts = attr_path.split(".")
            for part in parts[:-1]:
                self.saved.setdefault(attr_path, getattr(obj, part))
                obj = getattr(obj, part)
            self.saved[attr_path] = getattr(obj, parts[-1], None)
            setattr(obj, parts[-1], value)
        return Config

    def __exit__(self, *args):
        for attr_path, value in self.saved.items():
            obj = Config
            parts = attr_path.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)


# ==================================================================
# Single trial run
# ==================================================================

def make_env() -> VecNormalize:
    # Create a single VecNormalize-wrapped TORCS environment.
    raw_env = DummyVecEnv([lambda: TorcsSACEnv(stage=1)])
    env = VecNormalize(
        raw_env,
        norm_obs=False,
        norm_reward=False,   # must be False — see train_sac.py make_env() for explanation
        clip_reward=10.0,
        gamma=Config.sac.gamma,
    )
    return env


def evaluate(model: BCAnchoredSAC, env: VecNormalize, n_episodes: int = 3) -> dict:
    # Run deterministic evaluation episodes and return lap statistics.
    #
    # Returns
    # -------
    # dict with keys: best_lap_time, avg_lap_time, n_laps, max_dist, n_crashes
    lap_times = []
    n_crashes = 0
    max_dists = []
    obs = env.reset()

    for ep in range(n_episodes):
        ep_max_dist = 0.0
        ep_last_lap = None
        terminated = False

        for _ in range(Config.torcs.max_steps_per_episode):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)

            dist = float(infos[0].get("distRaced", 0.0))
            ep_max_dist = max(ep_max_dist, dist)

            if infos[0].get("lap_completed", False):
                lap_time = float(infos[0].get("lastLapTime", 0.0))
                if lap_time > 0 and lap_time != ep_last_lap:
                    ep_last_lap = lap_time
                    lap_times.append(lap_time)

            if dones[0]:
                terminated = True
                break

        max_dists.append(ep_max_dist)
        if terminated and ep_max_dist < 100:
            n_crashes += 1

    best_lap = min(lap_times) if lap_times else float("inf")
    avg_lap = np.mean(lap_times) if lap_times else float("inf")

    return {
        "best_lap_time": best_lap,
        "avg_lap_time": avg_lap,
        "n_laps": len(lap_times),
        "max_dist": max(max_dists),
        "n_crashes": n_crashes,
    }


def sample_hyperparams(trial: optuna.Trial) -> dict:
    # Define the Optuna hyperparameter search space.
    #
    # Returns a dict mapping config attribute paths to suggested values.
    # NOTE: the network architecture is FIXED at [256, 256, 128] here, NOT searched.
    # The BC/DAgger policy we initialize from has that exact actor architecture, so a
    # different net would silently discard the pretrained weights (shape mismatch) and
    # train from scratch — defeating the whole point. Keep arch fixed for fine-tuning.
    #
    # The learning rate is kept LOW (≤ 3e-4) to fine-tune the competent DAgger policy
    # without catastrophically forgetting it. A high LR can wipe out the imitation init.
    return {
        # SAC core — deliberately GENTLE ranges so the BC policy cannot collapse fast.
        # Very low LR + 1 gradient step/env-step + tiny fixed entropy means the actor
        # changes slowly, giving the (warm-started) critic time to learn that not-moving
        # and crashing are bad before the actor can drift there. We trade exploration
        # speed for not destroying the imitation init — once a trial holds the BC distance
        # (~2500m) instead of collapsing to 0m, we widen these ranges again.
        "sac.learning_rate": trial.suggest_float("lr", 5e-6, 5e-5, log=True),
        "sac.batch_size": trial.suggest_categorical("batch_size", [256, 512]),
        "sac.gamma": trial.suggest_float("gamma", 0.98, 0.995),
        "sac.tau": trial.suggest_float("tau", 0.001, 0.01),
        "sac.gradient_steps": 1,  # fixed at 1 — fewer updates per step = gentler
        # Tiny fixed entropy — just enough exploration, not enough to randomize the actor.
        "sac.ent_coef": trial.suggest_float("ent_coef", 0.001, 0.02, log=True),

        # Reward weights
        "reward.w_progress": trial.suggest_float("w_progress", 0.5, 5.0),
        "reward.w_time": trial.suggest_float("w_time", 0.01, 0.5),
        "reward.w_speed_bonus": trial.suggest_float("w_speed_bonus", 0.0, 2.0),
        "reward.w_cornering": trial.suggest_float("w_cornering", 0.0, 3.0),
        "reward.lap_bonus": trial.suggest_float("lap_bonus", 100, 1000),
        "reward.w_trackpos": trial.suggest_float("w_trackpos", -0.5, 0.0),
        "reward.w_angle": trial.suggest_float("w_angle", -0.5, 0.0),
        "reward.w_smoothness": trial.suggest_float("w_smoothness", -0.5, 0.0),

        # Driver aids (tcs only — steer rate limiter is disabled during training)
        "action.tcs_slip_threshold": trial.suggest_float("tcs_slip", 2.0, 15.0),
    }


def run_trial(
    trial: optuna.Trial,
    bc_pretrain_path: str = None,
    device: str = "auto",
    n_train_steps: int = 500_000,
    prune_after: int = 200_000,
    controller: str = "v2",
    teacher_params_path: str = None,
    seed_demos: int = 20_000,
    verbose: int = 0,
) -> float:
    # Run a single Optuna trial: apply hyperparams, train, evaluate.
    #
    # Parameters
    # ----------
    # trial : optuna.Trial
    # Current trial (used for reporting and pruning).
    # bc_pretrain_path : str or None
    # Path to BC-pretrained weights for actor initialization.
    # device : str
    # 'auto', 'cuda', or 'cpu'.
    # n_train_steps : int
    # Number of training steps for this trial.
    # prune_after : int
    # Step at which to check for pruning (no lap completed = prune).
    # verbose : int
    # Verbosity level.
    #
    # Returns
    # -------
    # float
    # Best lap time in seconds (lower is better). float('inf') if no lap.
    # Sample hyperparameters
    params = sample_hyperparams(trial)

    # Apply to Config
    with TempConfig(params):
        sac_cfg = Config.sac
        resolved_device = device if device != "auto" else Config.get_device()

        # Create environment
        env = make_env()

        # Create model with trial hyperparameters using BCAnchoredSAC.
        # bc_coef0 / bc_decay_steps are fixed (not searched) — they are architecture
        # choices, not SAC hyperparameters.  The anchor prevents policy collapse
        # regardless of which learning-rate / gamma Optuna is testing.
        model = BCAnchoredSAC(
            policy=LayerNormSACPolicy,
            env=env,
            learning_rate=sac_cfg.learning_rate,
            buffer_size=sac_cfg.buffer_size,
            batch_size=sac_cfg.batch_size,
            tau=sac_cfg.tau,
            gamma=sac_cfg.gamma,
            train_freq=(sac_cfg.train_freq, "step"),
            gradient_steps=sac_cfg.gradient_steps,
            ent_coef=sac_cfg.ent_coef,
            target_entropy=sac_cfg.target_entropy,
            learning_starts=0,
            policy_kwargs=sac_cfg.policy_kwargs,
            verbose=0,
            device=resolved_device,
            bc_coef0=sac_cfg.bc_coef0,
            bc_decay_steps=sac_cfg.bc_decay_steps,
        )

        # Load BC/DAgger pretrained weights (all 4 bugs from S3 are fixed in load_bc_weights)
        if bc_pretrain_path and os.path.exists(bc_pretrain_path):
            from training.train_sac import load_bc_weights
            load_bc_weights(model, bc_pretrain_path, verbose=0)
            model.learning_starts = 0

        # Seed replay buffer + automatically populate model.demo_obs/acts for BC anchor
        # (seed_replay_buffer calls model.add_demo_data() when model is BCAnchoredSAC)
        if seed_demos > 0:
            from training.train_sac import seed_replay_buffer
            from agents.bc_pretrain import build_teacher
            teacher = build_teacher(controller, teacher_params_path)
            seed_replay_buffer(model, env, n_steps=seed_demos, teacher=teacher, verbose=0)
            model.learning_starts = 0

        # FreezeActorCallback: freezes actor for the first freeze_steps env steps so the
        # critic can learn from seeded data before the actor receives any gradients.
        # Created ONCE and passed to every model.learn() chunk so its _frozen state
        # persists across chunks (actor only frozen during the very first block).
        freeze_cb = FreezeActorCallback(
            freeze_steps=Config.sac.freeze_steps, verbose=0
        )

        # Training loop with pruning check.
        best_lap = float("inf")
        checkpoint_interval = max(1, min(prune_after // 5, 20_000))

        for step_start in range(0, n_train_steps, checkpoint_interval):
            steps_this_block = min(checkpoint_interval, n_train_steps - step_start)

            model.learn(
                total_timesteps=steps_this_block,
                reset_num_timesteps=(step_start == 0),
                tb_log_name=f"optuna_trial_{trial.number}",
                callback=freeze_cb,
                progress_bar=False,
            )

            # Evaluate
            stats = evaluate(model, env, n_episodes=2)
            current_best = stats["best_lap_time"]

            if current_best < best_lap:
                best_lap = current_best

            # Per-block progress so you can SEE the policy improving in real time:
            # max_dist climbing toward 3602m means it's getting closer to a full lap.
            lap_str = f"{current_best:.2f}s" if current_best < float("inf") else "no lap yet"
            print(f"  [Trial {trial.number}] {step_start + steps_this_block:,} steps | "
                  f"max_dist={stats['max_dist']:.0f}m / 3602m | best_lap={lap_str}")

            trial.report(current_best, step_start + steps_this_block)

            # Prune if no lap completed by prune_after steps
            if step_start + steps_this_block >= prune_after:
                if current_best == float("inf"):
                    raise optuna.TrialPruned(
                        f"No lap completed at step {step_start + steps_this_block}"
                    )

            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Final evaluation
        final_stats = evaluate(model, env, n_episodes=3)
        if final_stats["best_lap_time"] < best_lap:
            best_lap = final_stats["best_lap_time"]

        if best_lap == float("inf") and final_stats["max_dist"] > 0:
            best_lap = 999.0  # completed distance but no lap = very slow

        env.close()

    return best_lap


# ==================================================================
# Optuna study management
# ==================================================================

def create_study(storage, study_name: str, n_trials: int):
    # Create or load an Optuna study and print its status.
    study = optuna.create_study(
        storage=_make_storage(storage),
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(n_startup_trials=20, multivariate=True),
        pruner=MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=100_000,
            n_min_trials=5,
        ),
        load_if_exists=True,
    )
    print(f"\n{'='*55}")
    print(f"  Optuna Study: {study_name}")
    print(f"  Storage:      {storage}")
    print(f"  Direction:    minimize (lap time in seconds)")
    print(f"  Sampler:      TPE (multivariate)")
    print(f"  Pruner:       MedianPruner (warmup=100K steps)")
    print(f"  Trials:       {len(study.trials)} completed so far")
    print(f"  Target:       {n_trials} total trials")
    print(f"{'='*55}\n")
    return study


def run_worker(
    storage: str,
    study_name: str,
    bc_pretrain_path: str = None,
    device: str = "auto",
    n_train_steps: int = 500_000,
    n_concurrent_trials: int = 1,
    target_trials: int = 200,
    controller: str = "v2",
    teacher_params_path: str = None,
    seed_demos: int = 20_000,
    verbose: bool = True,
):
    # Run worker that pulls trials from the study and executes them.
    #
    # Parameters
    # ----------
    # storage : str
    # SQL storage URL.
    # study_name : str
    # Name of the Optuna study.
    # bc_pretrain_path : str or None
    # Path to BC-pretrained weights.
    # device : str
    # Training device.
    # n_train_steps : int
    # Steps per trial.
    # n_concurrent_trials : int
    # How many trials to run on this worker (default 1).
    # Set >1 if the worker has multiple GPUs.
    # verbose : bool
    study = optuna.load_study(
        storage=_make_storage(storage),
        study_name=study_name,
    )

    print(f"\n[Worker] Connected to study '{study_name}'")
    print(f"[Worker] Device: {device}")
    if bc_pretrain_path:
        print(f"[Worker] BC weights: {bc_pretrain_path}")
    print(f"[Worker] Steps per trial: {n_train_steps:,}")
    print(f"[Worker] Concurrent trials: {n_concurrent_trials}")
    print()

    trial_count = 0
    start_time = time.time()

    # Run trials continuously until the study reaches target_trials or the user
    # interrupts. This keeps a beast machine busy all night instead of doing one
    # trial and quitting.
    try:
        while True:
            n_done = len([t for t in study.trials
                          if t.state in (optuna.trial.TrialState.COMPLETE,
                                         optuna.trial.TrialState.PRUNED)])
            if n_done >= target_trials:
                print(f"[Worker] Study reached target of {target_trials} trials. Stopping.")
                break

            trial = study.ask()
            trial_num = trial.number
            print(f"[Worker] Starting trial {trial_num} ({n_done}/{target_trials} done)...")

            t0 = time.time()
            try:
                best_lap = run_trial(
                    trial=trial,
                    bc_pretrain_path=bc_pretrain_path,
                    device=device,
                    n_train_steps=n_train_steps,
                    controller=controller,
                    teacher_params_path=teacher_params_path,
                    seed_demos=seed_demos,
                    verbose=1 if verbose else 0,
                )
                study.tell(trial, best_lap)
                elapsed = time.time() - t0
                if best_lap < float("inf"):
                    print(f"[Worker] Trial {trial_num} complete: "
                          f"best_lap={best_lap:.2f}s ({elapsed:.0f}s)")
                else:
                    print(f"[Worker] Trial {trial_num} complete: "
                          f"no lap completed ({elapsed:.0f}s)")
            except optuna.TrialPruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                print(f"[Worker] Trial {trial_num} pruned at {time.time()-t0:.0f}s")

            trial_count += 1

    except KeyboardInterrupt:
        print("\n[Worker] Stopped by user (Ctrl+C). All completed trials are saved in the DB. "
              "Re-run the same command anytime to continue.")

    total_elapsed = time.time() - start_time
    print(f"\n[Worker] Completed {trial_count} trials in {total_elapsed:.0f}s")
    print(f"[Worker] Average: {total_elapsed / max(1, trial_count):.0f}s per trial")
    print(f"[Worker] Study '{study_name}' has {len(study.trials)} total trials.\n")


def print_report(storage: str, study_name: str):
    # Print the current best trials and hyperparameters.
    study = optuna.load_study(
        storage=_make_storage(storage),
        study_name=study_name,
    )

    print(f"\n{'='*55}")
    print(f"  Optuna Report: {study_name}")
    print(f"  Trials: {len(study.trials)} total, "
          f"{len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])} complete, "
          f"{len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])} pruned")
    print(f"{'='*55}")

    # Gather completed trials with a value. Guard against "no completed trials yet"
    # (study.best_trial raises ValueError when nothing has finished).
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not trials:
        running = len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])
        print(f"\n  No completed trials yet ({running} currently running).")
        print(f"  Each trial takes ~70-90 min — let at least one finish, then check again.")
        return

    trials.sort(key=lambda t: t.value)
    best_trial = trials[0]
    print(f"\n  Best trial (#{best_trial.number}):")
    print(f"    Best lap time: {best_trial.value:.2f}s")
    print(f"    Params:")
    for key, value in best_trial.params.items():
        print(f"      {key}: {value}")

    print(f"\n  Top 5 trials:")
    for t in trials[:5]:
        print(f"    #{t.number}: {t.value:.2f}s")

    # Save report to JSON
    report_path = os.path.join(
        Config.CHECKPOINT_DIR,
        f"optuna_report_{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    report = {
        "study_name": study_name,
        "n_trials": len(study.trials),
        "n_complete": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
        "n_pruned": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
        "best_trial": {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
        },
        "top_5": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in trials[:5]
        ],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {report_path}")
    print()


# ==================================================================
# Main
# ==================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Distributed Optuna hyperparameter tuning for SAC TORCS Racing"
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["coordinator", "worker", "report"],
        help="Mode: 'coordinator' (creates study, then exits), "
             "'worker' (runs trials), 'report' (prints results)"
    )
    parser.add_argument(
        "--storage", type=str,
        default="sqlite:///optuna_study.db",
        help="SQL storage URL. For distributed: "
             "'mysql://user:pass@host/db' or 'postgresql://...'. "
             "Default: sqlite:///optuna_study.db (single machine only)"
    )
    parser.add_argument(
        "--study-name", type=str, default="sac_torcs_v1",
        help="Name of the Optuna study (default: sac_torcs_v1)"
    )
    parser.add_argument(
        "--n-trials", type=int, default=200,
        help="Total number of trials for coordinator (default: 200)"
    )
    parser.add_argument(
        "--n-concurrent", type=int, default=1,
        help="Number of concurrent trials per worker (default: 1)"
    )
    parser.add_argument(
        "--n-train-steps", type=int, default=500_000,
        help="Training steps per trial (default: 500000)"
    )
    parser.add_argument(
        "--bc-pretrain", type=str, default=None,
        help="Path to BC-pretrained .pth weights (default: uses pre-generated "
             "checkpoints/bc_pretrained.pth if exists)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device (default: auto)"
    )
    parser.add_argument(
        "--controller", type=str, default="v2", choices=["v1", "v2", "v3"],
        help="Teacher controller used to seed the replay buffer (default: v2)."
    )
    parser.add_argument(
        "--teacher-params", type=str, default=None,
        help="Path to tuned teacher params JSON for buffer seeding "
             "(e.g. checkpoints/best_teacher_params.json)."
    )
    parser.add_argument(
        "--seed-demos", type=int, default=20_000,
        help="Teacher transitions to seed each trial's replay buffer (default: 20000). "
             "This gives SAC clean full-lap data + the lap bonus. Set 0 to disable."
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Verbose output"
    )
    args = parser.parse_args()

    # Auto-detect pretrained actor: prefer the DAgger policy (most robust), then BC.
    bc_path = args.bc_pretrain
    if bc_path is None:
        for cand in ("dagger_policy.pth", "bc_pretrained.pth"):
            p = os.path.join(Config.CHECKPOINT_DIR, cand)
            if os.path.exists(p):
                bc_path = p
                break
    if bc_path:
        print(f"[Optuna-SAC] Actor will be initialized from: {bc_path}")

    if args.mode == "coordinator":
        study = create_study(args.storage, args.study_name, args.n_trials)
        print(f"Study created/loaded. Run workers to begin trials:")
        print(f"  python optuna_sac.py --mode worker --storage \"{args.storage}\" "
              f"--study-name \"{args.study_name}\" --device cuda")

    elif args.mode == "worker":
        run_worker(
            storage=args.storage,
            study_name=args.study_name,
            bc_pretrain_path=bc_path,
            device=args.device,
            n_train_steps=args.n_train_steps,
            n_concurrent_trials=args.n_concurrent,
            target_trials=args.n_trials,
            controller=args.controller,
            teacher_params_path=args.teacher_params,
            seed_demos=args.seed_demos,
            verbose=args.verbose,
        )

    elif args.mode == "report":
        print_report(args.storage, args.study_name)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

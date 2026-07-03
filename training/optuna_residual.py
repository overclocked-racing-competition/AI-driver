# Distributed Optuna HPO over residual-SAC hyperparameters (delta, LR, ent_coef,
# log_std_init). Each trial: ResidualTorcsEnv on the frozen base, residual=0
# seeding, short training, deterministic lap-time objective (minimize).
# Modes: --mode coordinator | worker | report.

import os
import sys
import time
import argparse
import warnings
from types import SimpleNamespace

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
# Reuse the distributed infra from optuna_sac (generic — storage/study/report)
from training.optuna_sac import _make_storage, TempConfig, create_study, print_report
# Reuse the TESTED residual functions
from training.train_residual_sac import (
    make_env, create_model, seed_replay_buffer, evaluate_learned,
)
from core.callbacks import FreezeActorCallback

DAGGER_BASELINE_LAP = 106.96   # seconds — what residual RL must beat


# ==================================================================
# Search space
# ==================================================================

# Search space — single source of truth for both the sampler and the warm-start
# importer, so their distributions can't drift (drift caused the "different log
# configuration to the same parameter name" crash). v2 widened delta and ent_coef
# ranges after v1's best trial railed against the old maxima.
SEARCH_DISTRIBUTIONS = {
    # Correction bound — how far SAC may deviate from DAgger per action dim.
    # Too small: can't improve. Too large: exploration crashes the car.
    "delta":        optuna.distributions.FloatDistribution(0.05, 0.40),
    # Actor LR — how fast the residual moves. Low keeps it gentle. (LOG scale)
    "lr":           optuna.distributions.FloatDistribution(3e-5, 3e-4, log=True),
    # Entropy coefficient — exploration pressure. (LOG scale)
    "ent_coef":     optuna.distributions.FloatDistribution(0.005, 0.12, log=True),
    # gSDE exploration scale. Lower = gentler (gSDE scales noise by latent magnitude).
    "log_std_init": optuna.distributions.FloatDistribution(-5.0, -2.0),
    # Discount + target update
    "gamma":        optuna.distributions.FloatDistribution(0.98, 0.995),
    "tau":          optuna.distributions.FloatDistribution(0.002, 0.01),
}

# Optuna param name → Config.residual.* attribute path (for TempConfig)
_PARAM_TO_CONFIG = {
    "delta":        "residual.delta",
    "lr":           "residual.learning_rate",
    "ent_coef":     "residual.ent_coef",
    "log_std_init": "residual.log_std_init",
    "gamma":        "residual.gamma",
    "tau":          "residual.tau",
}


def sample_hyperparams(trial: optuna.Trial) -> dict:
    # Residual RL search space — maps Optuna params to Config.residual.* paths.
    #
    # Reads ranges from SEARCH_DISTRIBUTIONS (the single source of truth shared
    # with seed_v2_from_v1) so the live search and any warm-start import always
    # declare identical distributions.
    out = {}
    for name, dist in SEARCH_DISTRIBUTIONS.items():
        value = trial.suggest_float(name, dist.low, dist.high, log=dist.log)
        out[_PARAM_TO_CONFIG[name]] = value
    return out


# ==================================================================
# Single trial
# ==================================================================

def run_trial(
    trial: optuna.Trial,
    device: str = "auto",
    n_train_steps: int = 200_000,
    eval_interval: int = 100_000,
    verbose: int = 0,
) -> float:
    # Run one residual-RL trial. Returns the best deterministic lap time (seconds).
    # Lower is better. A trial that never laps returns a large distance-scaled penalty.
    params = sample_hyperparams(trial)

    with TempConfig(params):
        resolved_device = device if device != "auto" else Config.get_device()
        # Minimal args object for the reused train_residual_sac functions.
        # lr/delta/dagger_weights = None → functions fall back to (overridden) Config.residual.
        args = SimpleNamespace(
            device=resolved_device, verbose=verbose,
            lr=None, delta=None, dagger_weights=None,
        )

        env = make_env()                 # uses Config.residual.delta (overridden)
        model = create_model(env, args)  # gentle init w/ Config.residual.log_std_init (overridden)

        # Seed with DAgger baseline (residual=0) — gives the critic full-lap data
        seed_replay_buffer(model, env, n_steps=Config.residual.seed_steps, verbose=verbose)

        freeze_cb = FreezeActorCallback(
            freeze_steps=Config.residual.freeze_steps, verbose=0
        )

        best_lap = float("inf")
        best_dist = 0.0
        step_done = 0

        try:
            while step_done < n_train_steps:
                chunk = min(eval_interval, n_train_steps - step_done)
                model.learn(
                    total_timesteps=chunk,
                    callback=freeze_cb,
                    reset_num_timesteps=(step_done == 0),
                    tb_log_name=f"residual_trial_{trial.number}",
                    progress_bar=False,
                )
                step_done += chunk

                # Evaluate the DETERMINISTIC learned policy (the real objective)
                avg_dist, lap = evaluate_learned(
                    model, args, n_episodes=2, step_label=f"@{step_done//1000}k"
                )
                best_dist = max(best_dist, avg_dist)
                if lap < best_lap:
                    best_lap = lap

                # Report to Optuna for pruning (use lap time, or a big number if no lap)
                report_val = best_lap if best_lap < float("inf") else 999.0
                trial.report(report_val, step_done)

                print(f"  [Trial {trial.number}] {step_done:,} steps | "
                      f"best_lap={'%.2f' % best_lap if best_lap < float('inf') else 'no lap'} | "
                      f"avg_dist={avg_dist:.0f}m")

                if trial.should_prune():
                    env.close()
                    raise optuna.TrialPruned()

        finally:
            try:
                env.close()
            except Exception:
                pass

    # Objective: lap time (lower better). If it never lapped, return a penalty that
    # still rewards partial progress (so Optuna can climb out of the no-lap region).
    if best_lap < float("inf"):
        return best_lap
    return 999.0 - best_dist / 100.0   # e.g. 2000m → 979 (better than 0m → 999)


# ==================================================================
# Worker loop
# ==================================================================

def run_worker(storage, study_name, device, n_train_steps, target_trials, verbose):
    study = optuna.load_study(storage=_make_storage(storage), study_name=study_name)
    print(f"\n[Worker] Connected to '{study_name}' | device={device} | "
          f"{n_train_steps:,} steps/trial")

    count = 0
    t_start = time.time()
    try:
        while True:
            n_done = len([t for t in study.trials
                          if t.state in (optuna.trial.TrialState.COMPLETE,
                                         optuna.trial.TrialState.PRUNED)])
            if n_done >= target_trials:
                print(f"[Worker] Study reached {target_trials} trials. Stopping.")
                break

            trial = study.ask()
            print(f"\n[Worker] Trial {trial.number} starting ({n_done}/{target_trials} done)...")
            t0 = time.time()
            try:
                lap = run_trial(trial, device=device, n_train_steps=n_train_steps,
                                verbose=1 if verbose else 0)
                study.tell(trial, lap)
                tag = (f"BEAT DAgger by {DAGGER_BASELINE_LAP - lap:.2f}s!"
                       if lap < DAGGER_BASELINE_LAP else f"lap {lap:.2f}s")
                print(f"[Worker] Trial {trial.number} done: {tag} ({time.time()-t0:.0f}s)")
            except optuna.TrialPruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                print(f"[Worker] Trial {trial.number} pruned ({time.time()-t0:.0f}s)")
            count += 1
    except KeyboardInterrupt:
        print("\n[Worker] Stopped by user. All completed trials are saved in the DB.")

    print(f"\n[Worker] Ran {count} trials in {time.time()-t_start:.0f}s")


# ==================================================================
# Main
# ==================================================================

def print_best_command(storage: str, study_name: str, timesteps: int = 2_000_000):
    # Print the exact train_residual_sac.py command to reproduce the best trial
    # from this study for a full long training run. Avoids hand-copying floats.
    study = optuna.load_study(storage=_make_storage(storage), study_name=study_name)
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not trials:
        print("[BestCommand] No completed trials yet.")
        return

    best = min(trials, key=lambda t: t.value)
    p = best.params
    cmd = (
        f"D:\\torcs\\pyenv\\Scripts\\python.exe train_residual_sac.py"
        f" --timesteps {timesteps}"
        f" --resume \"\""
        f" --delta {p['delta']:.6f}"
        f" --lr {p['lr']:.2e}"
        f" --ent-coef {p['ent_coef']:.6f}"
        f" --log-std-init {p['log_std_init']:.4f}"
        f" --gamma {p['gamma']:.6f}"
        f" --tau {p['tau']:.6f}"
    )
    print(f"\n[BestCommand] Best trial #{best.number} = {best.value:.2f}s")
    print(f"[BestCommand] Paste this on the dedicated training machine:\n")
    print(f"  {cmd}\n")


def seed_v2_from_v1(
    storage: str,
    source_study: str,
    target_study: str,
    n_trials: int = 100,
):
    # Warm-start a new (wider) study by importing all COMPLETE trials from a
    # previous study. TPE then starts informed rather than cold.
    #
    # All v1 param values fit inside v2's wider ranges (v2 only widened the max,
    # never narrowed), so every imported trial is a valid v2 data point.
    st = _make_storage(storage)
    src = optuna.load_study(storage=st, study_name=source_study)
    dst = optuna.load_study(storage=st, study_name=target_study)

    names = list(SEARCH_DISTRIBUTIONS.keys())
    complete = [t for t in src.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    print(f"\n[SeedV2] Importing up to {len(complete)} completed trials "
          f"from '{source_study}' -> '{target_study}'")

    new_trials, skipped = [], 0
    for t in complete:
        # Need all 6 params present and each within the v2 distribution range.
        # (v2 only widened maxima, so all v1 values should fit — but check to be safe.)
        if not all(n in t.params for n in names):
            skipped += 1
            continue
        if not all(SEARCH_DISTRIBUTIONS[n].low <= t.params[n] <= SEARCH_DISTRIBUTIONS[n].high
                   for n in names):
            skipped += 1
            continue
        # create_trial with distributions that EXACTLY match the live sampler — this is
        # what avoids the "different log configuration" conflict. Same dict, same log flags.
        new_trials.append(optuna.trial.create_trial(
            params={n: t.params[n] for n in names},
            distributions=dict(SEARCH_DISTRIBUTIONS),
            value=t.value,
        ))

    if new_trials:
        dst.add_trials(new_trials)

    print(f"[SeedV2] Imported {len(new_trials)} trials ({skipped} skipped).")
    print(f"[SeedV2] '{target_study}' now has {len(dst.trials)} trials total.")
    if len(dst.trials) > 0:
        try:
            best = dst.best_trial
            print(f"[SeedV2] Current best in '{target_study}': #{best.number} = {best.value:.2f}s")
        except ValueError:
            pass


def main():
    p = argparse.ArgumentParser(description="Distributed Optuna for Residual RL")
    p.add_argument("--mode", required=True,
                   choices=["coordinator", "worker", "report", "best-command", "seed-v2-from-v1"])
    p.add_argument("--storage", default="sqlite:///optuna_residual.db",
                   help="SQL storage URL (postgresql://... for distributed)")
    p.add_argument("--study-name", default="residual_v1")
    p.add_argument("--source-study", default="residual_v1",
                   help="Source study name for seed-v2-from-v1 (default: residual_v1)")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--n-train-steps", type=int, default=200_000,
                   help="Training steps per trial (default 200k)")
    p.add_argument("--timesteps", type=int, default=2_000_000,
                   help="Full training timesteps for best-command output (default 2M)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--verbose", action="store_true", default=False)
    args = p.parse_args()

    if args.mode == "coordinator":
        create_study(args.storage, args.study_name, args.n_trials)
        print(f"\nStudy '{args.study_name}' created. Next steps:")
        print(f"  1. Seed from previous study:  python optuna_residual.py --mode seed-v2-from-v1 "
              f'--storage "{args.storage}" --study-name "{args.study_name}" --source-study residual_v1')
        print(f"  2. Start workers:              python optuna_residual.py --mode worker "
              f'--storage "{args.storage}" --study-name "{args.study_name}" --device cuda')

    elif args.mode == "worker":
        run_worker(args.storage, args.study_name, args.device,
                   args.n_train_steps, args.n_trials, args.verbose)

    elif args.mode == "report":
        print_report(args.storage, args.study_name)

    elif args.mode == "best-command":
        print_best_command(args.storage, args.study_name, timesteps=args.timesteps)

    elif args.mode == "seed-v2-from-v1":
        seed_v2_from_v1(args.storage, args.source_study, args.study_name, args.n_trials)


if __name__ == "__main__":
    main()

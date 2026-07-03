# Create + seed the v2 lookahead-PID study (legacy).
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import TrialState

from search.optuna_teacher_v3 import make_storage

ap = argparse.ArgumentParser()
ap.add_argument("--study-name", default="teacher_v2_ow1")
ap.add_argument("--storage", default=os.environ.get("OPTUNA_STORAGE"))
args = ap.parse_args()
if not args.storage:
    print("ERROR: no --storage and $OPTUNA_STORAGE unset."); sys.exit(1)

study = optuna.create_study(
    study_name=args.study_name, storage=make_storage(args.storage), direction="minimize",
    sampler=TPESampler(n_startup_trials=40, multivariate=True, group=True, seed=42),
    pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=1, interval_steps=1),
    load_if_exists=True,
)
done = len([t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)])
print(f"[study] '{args.study_name}' ready. completed: {done}")

# Short optuna param names must match teacher_controller_v2.sample_params().
BASE = dict(steer_gain=26.0, centering_gain=0.18, steer_damp=0.25, rl_entry=-0.30, rl_exit=0.20,
            entry_thr=60.0, exit_thr=120.0, max_speed=260.0, base_corner_spd=45.0, speed_per_m=1.25,
            accel_gain=2.0, brake_gain=2.5, trail_floor=0.10, abs_slip=3.0, abs_min_spd=20.0,
            abs_rel=0.70, tcs_slip=6.0, tcs_min_spd=25.0, tcs_min_fac=0.15, launch_steps=12,
            rpm_up=8200.0, rpm_down=3500.0, gear_scale=0.85)

def variant(**over):
    d = dict(BASE); d.update(over); return d

SEEDS = [
    BASE,
    variant(max_speed=285.0, speed_per_m=1.7, accel_gain=3.2, brake_gain=3.8,
            base_corner_spd=52.0, gear_scale=0.72),   # aggressive
    variant(max_speed=275.0, speed_per_m=1.5, accel_gain=2.8, brake_gain=3.3,
            base_corner_spd=48.0, gear_scale=0.78, rl_entry=-0.40, rl_exit=0.30),  # mid + more line
]

if done == 0:
    for s in SEEDS:
        study.enqueue_trial(s, skip_if_exists=False)
    print(f"[study] enqueued {len(SEEDS)} v2 seeds.")
else:
    print("[study] already has trials — skipped seeding.")
print("[study] done.")

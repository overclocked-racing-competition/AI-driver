# Create + seed the v4 physics-teacher study (legacy).
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import TrialState

from search.optuna_teacher_v3 import make_storage

ap = argparse.ArgumentParser()
ap.add_argument("--study-name", default="teacher_v4_ow1")
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

# Param names == teacher_controller_v4.sample_params() (dataclass field names).
BASE = dict(max_speed=305.0, corner_coef=18.0, margin_m=5.0, min_speed=42.0,
            accel_gain=3.2, brake_gain=4.2, trail_throttle_floor=0.07,
            k_aim=0.5, k_angle=13.0, k_center=0.20, steer_damp=0.20,
            abs_slip_threshold=3.0, abs_release_fraction=0.70, tcs_slip_threshold=6.0,
            launch_steps=12, rpm_upshift=8300.0, gear_speed_scale=0.80)

def variant(**over):
    d = dict(BASE); d.update(over); return d

SEEDS = [
    BASE,
    variant(corner_coef=22.0, max_speed=315.0, k_aim=0.7, brake_gain=5.0, margin_m=3.0),   # aggressive
    variant(corner_coef=15.0, k_aim=0.3, k_center=0.28, margin_m=9.0, brake_gain=4.8),     # safer line
]

if done == 0:
    for s in SEEDS:
        study.enqueue_trial(s, skip_if_exists=False)
    print(f"[study] enqueued {len(SEEDS)} v4 seeds.")
else:
    print("[study] already has trials — skipped seeding.")
print("[study] done.")

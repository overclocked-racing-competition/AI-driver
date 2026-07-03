# new_study_v5.py — create + seed the V5 predictive-braking study.
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import TrialState
from search.optuna_teacher_v3 import make_storage

ap = argparse.ArgumentParser()
ap.add_argument("--study-name", default="teacher_v5_ow1")
ap.add_argument("--storage", default=os.environ.get("OPTUNA_STORAGE"))
a = ap.parse_args()
if not a.storage:
    print("ERROR: no --storage / $OPTUNA_STORAGE"); sys.exit(1)

study = optuna.create_study(
    study_name=a.study_name, storage=make_storage(a.storage), direction="minimize",
    sampler=TPESampler(n_startup_trials=40, multivariate=True, group=True, seed=42),
    pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=1, interval_steps=1),
    load_if_exists=True)
done = len([t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)])
print(f"[study] '{a.study_name}' ready. completed: {done}")

BASE = dict(max_speed=310.0, corner_coef=15.0, brake_reach=450.0, margin_m=5.0, min_speed=40.0,
            accel_gain=3.4, brake_gain=5.0, trail_throttle_floor=0.06, k_aim=0.5, k_angle=13.0,
            k_center=0.20, steer_damp=0.20, abs_slip_threshold=3.0, abs_release_fraction=0.70,
            tcs_slip_threshold=6.0, launch_steps=12, rpm_upshift=17800.0, rpm_downshift=9000.0)

def v(**o):
    d = dict(BASE); d.update(o); return d

SEEDS = [
    BASE,
    v(brake_reach=900.0, corner_coef=13.0, max_speed=320.0, brake_gain=6.5),     # bold late-braking
    v(brake_reach=250.0, corner_coef=13.0, margin_m=10.0, k_aim=0.3, k_center=0.28),  # safer
]
if done == 0:
    for s in SEEDS:
        study.enqueue_trial(s, skip_if_exists=False)
    print(f"[study] enqueued {len(SEEDS)} v5 seeds.")
else:
    print("[study] already has trials — skipped seeding.")
print("[study] done.")

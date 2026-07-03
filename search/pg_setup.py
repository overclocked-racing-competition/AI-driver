# Create (or load) the teacher_wp Optuna study in PostgreSQL (cloud) and seed it.
# Reads the connection URL from the OPTUNA_STORAGE env var so the password never
# appears on a command line or in logs. Run once when switching to Postgres.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import TrialState

from search.optuna_teacher_v3 import make_storage, SPEED_SEARCH, TRACKPOS_SEARCH

url = os.environ.get("OPTUNA_STORAGE")
if not url:
    print("ERROR: OPTUNA_STORAGE env var not set in this process.")
    sys.exit(1)
if "postgres" not in url:
    print("ERROR: OPTUNA_STORAGE is not a postgres URL.")
    sys.exit(1)

storage = make_storage(url)              # connects lazily
study = optuna.create_study(
    study_name="teacher_wp", storage=storage, direction="minimize",
    sampler=TPESampler(n_startup_trials=40, multivariate=True, group=True, seed=42),
    pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=1, interval_steps=1),
    load_if_exists=True,
)
done = len([t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)])
print(f"[pg] connected OK. study 'teacher_wp' ready in Postgres. completed trials: {done}")

# ---- Seed good param sets (lap-completing anchors) ----
tp_mid = {k: round((v[0]+v[1])/2, 3) for k, v in TRACKPOS_SEARCH.items()}
def mk(speeds, tsf, bg, bl, tc=0.40):
    p = dict(speeds)
    for k in TRACKPOS_SEARCH: p[k] = tp_mid[k]
    p.update(steer_angle_gain=27.0, steer_trackpos_gain=0.45, steer_damping=0.30,
             brake_gain=bg, brake_lookahead_m=bl, target_speed_factor=tsf,
             abs_slip_ratio=0.30, abs_brake_cut=0.35, throttle_in_corner=tc, accel_on_exit=0.80)
    return p

lo = {k: SPEED_SEARCH[k][0] for k in SPEED_SEARCH}                       # all range-low (laps ~189s)
A  = dict(speed_wp0=255, speed_wp1=88, speed_wp2=170, speed_wp3=70,      # fast straights / slow corkscrew
          speed_wp4=250, speed_wp5=250, speed_wp6=115, speed_wp7=80,
          speed_wp8=82,  speed_wp9=58, speed_wp10=95, speed_wp11=66)
B  = dict(speed_wp0=235, speed_wp1=85, speed_wp2=150, speed_wp3=68,
          speed_wp4=230, speed_wp5=230, speed_wp6=110, speed_wp7=78,
          speed_wp8=80,  speed_wp9=57, speed_wp10=90, speed_wp11=64)

if done == 0:   # only seed a fresh study
    for s in [mk(lo, 0.82, 2.8, 190), mk(A, 0.84, 2.5, 155), mk(B, 0.83, 2.5, 170)]:
        study.enqueue_trial(s, skip_if_exists=False)
    print("[pg] enqueued 3 seeds (safe + fast-straights + safer).")
else:
    print("[pg] study already has trials — skipped seeding.")
print("[pg] done.")

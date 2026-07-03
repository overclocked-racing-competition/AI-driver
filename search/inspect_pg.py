# Inspect the existing teacher_v3 study in Postgres: how many real laps, the best
# lap time + params, and which search-space ranges the old trials used (to detect a
# distribution conflict with the corrected ranges).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optuna
from optuna.trial import TrialState
from search.optuna_teacher_v3 import make_storage

study = optuna.load_study(study_name="teacher_v3", storage=make_storage(os.environ["OPTUNA_STORAGE"]))
complete = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
lapping  = [t for t in complete if t.value < 900.0]    # <900 => completed a lap (else 985-999 penalty)
print(f"total trials: {len(study.trials)} | complete: {len(complete)} | LAPPING (value<900): {len(lapping)}")

if lapping:
    lapping.sort(key=lambda t: t.value)
    print("\n=== 8 best lapping trials (value = best lap seconds) ===")
    for t in lapping[:8]:
        print(f"  trial {t.number}: {t.value:.3f}s")
    best = lapping[0]
    sp = {k: round(v, 1) for k, v in sorted(best.params.items()) if k.startswith('speed_wp')}
    print(f"\nBEST = {best.value:.3f}s (trial {best.number})")
    print("  speed waypoints:", sp)
else:
    print("\nNo lapping trials — all crashed (penalty values).")

# Detect which search space the OLD trials used (distribution range of speed_wp11).
if complete:
    d = complete[-1].distributions.get('speed_wp11')
    print(f"\nspeed_wp11 distribution in old trials: {d}")
    print("  (corrected range is 55..200; if old shows 200..290 -> distribution CONFLICT -> need a new study)")

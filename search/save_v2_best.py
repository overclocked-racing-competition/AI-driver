import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optuna
from optuna.trial import TrialState
from search.optuna_teacher_v3 import make_storage

study = optuna.load_study(study_name="teacher_v3", storage=make_storage(os.environ["OPTUNA_STORAGE"]))
lap = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None and t.value < 900]
lap.sort(key=lambda t: t.value)
best = lap[0]
d = dict(best.params)
d["_lap_time"] = best.value
d["_trial"] = best.number
d["_n_lapping_trials"] = len(lap)
d["_note"] = "v2 continuous-speed controller (incl. finish_zone). Reproducing 99.8s needs the matching teacher_controller_v2 version."
out = r"D:\IBM_competition\SAC\S3_B\S4-F\checkpoints\best_teacher_v2_pg_99.8s.json"
json.dump(d, open(out, "w"), indent=2)
print(f"saved {out}")
print(f"  best lap {best.value:.3f}s (trial {best.number}); {len(lap)} lapping trials; top values:",
      [round(t.value, 1) for t in lap[:6]])
# does the CURRENT teacher_controller_v2 have all these params?
import agents.teacher_controller_v2 as tc2
fields = set(tc2.TeacherParams.__dataclass_fields__.keys())
mapped = set(tc2._OPTUNA_TO_FIELD.get(k, k) for k in best.params.keys())
missing = mapped - fields
print(f"  params in PG best not in current teacher_controller_v2: {missing if missing else 'NONE (compatible!)'}")

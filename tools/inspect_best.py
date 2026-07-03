import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optuna
from optuna.trial import TrialState
from search.optuna_teacher_v3 import make_storage

study = optuna.load_study(study_name="teacher_v3", storage=make_storage(os.environ["OPTUNA_STORAGE"]))
lap = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None and t.value < 900]
lap.sort(key=lambda t: t.value)
best = lap[0]
print(f"BEST trial {best.number}: {best.value:.3f}s\n")
print("ALL params of best trial:")
for k, v in sorted(best.params.items()):
    print(f"  {k} = {v}")
print("\n--- param-name set (identifies the controller/search space) ---")
print(sorted(best.params.keys()))
# user attrs / system attrs may say which controller
print("\nuser_attrs:", best.user_attrs)
print("system_attrs keys:", list(best.system_attrs.keys()))

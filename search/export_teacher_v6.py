# Export best v6 Optuna trial params to JSON

import os
import sys
import json
import argparse
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from optuna.trial import TrialState
from search.optuna_teacher_v3 import make_storage
from agents.teacher_controller_v6 import params_from_optuna


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-name", default="teacher_v6_ow1")
    ap.add_argument("--storage",
                    default=os.environ.get("OPTUNA_STORAGE",
                                           "sqlite:////home/user/optuna_ow1.db"))
    ap.add_argument("--output", default="checkpoints/best_teacher_v6.json")
    a = ap.parse_args()

    study = optuna.load_study(study_name=a.study_name, storage=make_storage(a.storage))
    completed = [t for t in study.trials
                 if t.state == TrialState.COMPLETE
                 and t.value is not None and t.value < 900.0]
    if not completed:
        print("[export] No completed lapping trials to export.")
        sys.exit(1)

    best = min(completed, key=lambda t: t.value)
    params = params_from_optuna(best.params)
    d = asdict(params)

    out = a.output
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(d, f, indent=2)

    print(f"[export] best trial #{best.number}: {best.value:.3f}s "
          f"({len(completed)} lapping trials)")
    print(f"[export] {len(d)} params -> {out}")
    print(f"[export] next: python bc_pretrain.py --controller v6 --teacher-params {out}")


if __name__ == "__main__":
    main()

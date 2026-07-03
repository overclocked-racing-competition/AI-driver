# One-way mirror: COMPLETED trials from the local SQLite study to cloud Postgres,
# over a single PG connection (workers never touch the pooler). Dedup by local
# trial number, persisted to JSON. Cloud study must exist beforehand.
# Usage: python3 sync_to_cloud.py --study-name <name> --local sqlite:////path.db --interval 60
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from optuna.trial import TrialState, create_trial
from search.optuna_teacher_v3 import make_storage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-name", default="teacher_ow1")
    ap.add_argument("--local", required=True, help="local SQLite url, e.g. sqlite:////home/user/optuna_ow1.db")
    ap.add_argument("--cloud", default=os.environ.get("OPTUNA_STORAGE"), help="cloud PG url; default $OPTUNA_STORAGE")
    ap.add_argument("--interval", type=float, default=60.0)
    args = ap.parse_args()
    if not args.cloud:
        print("ERROR: no --cloud and $OPTUNA_STORAGE not set."); sys.exit(1)

    state_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              f".sync_{args.study_name}.json")
    try:
        synced = set(json.load(open(state_file)))
    except Exception:
        synced = set()

    local = optuna.load_study(study_name=args.study_name, storage=make_storage(args.local))
    cloud = optuna.load_study(study_name=args.study_name, storage=make_storage(args.cloud))
    print(f"[sync] local={args.local}  ->  cloud (PG)  |  already synced: {len(synced)}", flush=True)

    while True:
        try:
            pushed = skipped = 0
            for t in local.get_trials(deepcopy=False):
                if t.number in synced or t.state != TrialState.COMPLETE or t.value is None:
                    continue
                try:
                    cloud.add_trial(create_trial(
                        state=TrialState.COMPLETE, value=t.value,
                        params=t.params, distributions=t.distributions,
                        user_attrs={"local_trial": t.number}))
                    pushed += 1
                except Exception as e:
                    # e.g. a seed value outside the search distribution — skip that
                    # ONE trial permanently (mark synced) so it never blocks the rest.
                    print(f"[sync] skip local #{t.number}: {str(e)[:80]}", flush=True)
                    skipped += 1
                synced.add(t.number)
            if pushed or skipped:
                json.dump(sorted(synced), open(state_file, "w"))
            if pushed:
                best = min((t.value for t in cloud.get_trials(deepcopy=False)
                            if t.value is not None and t.value < 900.0), default=None)
                bs = f"{best:.3f}s" if best else "—"
                print(f"[sync] pushed {pushed} (skipped {skipped}) | local_done={len(synced)} | cloud_best={bs}", flush=True)
        except Exception as e:
            print(f"[sync] warn: {str(e)[:120]} — retry in {args.interval:.0f}s", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

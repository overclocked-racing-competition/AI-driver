# WSL Optuna worker for the v5 predictive late-braking teacher.
import os, sys, math, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from config import Config
from search.optuna_teacher_v3 import make_storage
from search.optuna_teacher_linux import launch_torcs, kill_torcs, install_practice_xml, TORCS_HOME  # noqa: F401
from agents.teacher_controller_v5 import TeacherController, sample_params

DAGGER = 106.96
INF = float("inf")


def evaluate(params, port, n_laps, timeout_s):
    import core.snakeoil3_gym as snakeoil3
    ctrl = TeacherController(params)
    laps, max_dist, t0 = [], 0.0, time.time()
    try:
        saved = sys.argv; sys.argv = [sys.argv[0]]
        client = snakeoil3.Client(p=port); sys.argv = saved
    except Exception:
        return {"best_lap": INF, "avg_lap": INF, "laps": 0, "max_dist": 0.0}
    ctrl.reset()
    client.MAX_STEPS = int(timeout_s * 50)
    last = None
    step = 0
    last_prog_step = 0
    last_dist = 0.0
    try:
        client.get_servers_input()
        while time.time() - t0 < timeout_s and len(laps) < n_laps:
            raw = client.S.d
            a = ctrl.act(raw)
            steer, ab = float(a[0]), float(a[1])
            accel, brake = (ab, 0.0) if ab >= 0 else (0.0, -ab)
            client.R.d["steer"] = steer; client.R.d["accel"] = accel
            client.R.d["brake"] = brake; client.R.d["gear"] = ctrl.get_gear()
            client.respond_to_server(); client.get_servers_input()
            raw = client.S.d
            step += 1
            dist = float(raw.get("distRaced", 0.0)); max_dist = max(max_dist, dist)
            llt = float(raw.get("lastLapTime", 0.0))
            if llt > 0 and llt != last:
                last = llt; laps.append(llt)
            # progress / stuck detection: bail if no forward distance for ~4s (car wedged
            # against a barrier while still "on track") instead of burning the full timeout.
            if dist > last_dist + 1.0:
                last_dist = dist; last_prog_step = step
            elif step - last_prog_step > 200:
                client.R.d["meta"] = True; client.respond_to_server(); break
            ang = float(raw.get("angle", 0.0)); tp = float(raw.get("trackPos", 0.0))
            dmg = float(raw.get("damage", 0.0))
            if (math.cos(ang) < Config.torcs.backwards_cos_threshold
                    or dmg > Config.torcs.max_damage
                    or abs(tp) > Config.torcs.offtrack_trackpos_threshold):
                client.R.d["meta"] = True; client.respond_to_server(); break
    except Exception:
        pass
    try:
        client.R.d["meta"] = True; client.respond_to_server()
    except Exception:
        pass
    return {"best_lap": min(laps) if laps else INF,
            "avg_lap": sum(laps)/len(laps) if laps else INF,
            "laps": len(laps), "max_dist": max_dist}


def obj(r):
    if r["laps"] > 0 and r["best_lap"] < INF:
        return r["best_lap"]
    return 999.0 - r["max_dist"] / (Config.torcs.track_length_m / 100.0)


def n_done(study):
    return len([t for t in study.trials if t.state in
                (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-name", default="teacher_v5_ow1")
    ap.add_argument("--storage", default=None)
    ap.add_argument("--n-trials", type=int, default=1000000)
    ap.add_argument("--n-laps", type=int, default=1)
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--timeout-s", type=float, default=200.0)
    ap.add_argument("--launch", choices=["r", "menu"], default="r")
    ap.add_argument("--display", default=":1")
    args = ap.parse_args()

    url = args.storage or os.environ.get("OPTUNA_STORAGE") or "sqlite:///optuna_teacher_v5.db"
    backend = "postgres" if "postgres" in url else ("sqlite" if "sqlite" in url else "?")
    study = optuna.load_study(study_name=args.study_name, storage=make_storage(url))
    race_xml = install_practice_xml(args.port)

    print(f"\n{'='*58}\n  V5 predictive-braking driver  [storage={backend}]")
    print(f"  Study: {args.study_name} | completed: {n_done(study)} | port={args.port}")
    print(f"{'='*58}\n", flush=True)

    t0, count, best, last_check = time.time(), 0, INF, 0
    try:
        gdone = n_done(study)
    except Exception:
        gdone = 0

    while gdone < args.n_trials:
        if count - last_check >= 25:
            try:
                gdone = n_done(study)
            except Exception:
                pass
            last_check = count
            if gdone >= args.n_trials:
                break
        try:
            trial = study.ask()
        except Exception as e:
            print(f"[warn] ask ({e}); 5s", flush=True); time.sleep(5); continue

        params = sample_params(trial)
        launch_torcs(race_xml, args.port, args.launch, args.display)
        try:
            r = evaluate(params, args.port, args.n_laps, args.timeout_s)
        except Exception as e:
            print(f"[trial {trial.number}] err {e}", flush=True)
            r = {"best_lap": INF, "avg_lap": INF, "laps": 0, "max_dist": 0.0}
        finally:
            kill_torcs()

        try:
            study.tell(trial, obj(r))
        except Exception:
            time.sleep(5)
            try:
                study.tell(trial, obj(r))
            except Exception:
                pass
        count += 1; gdone += 1
        if r["laps"] > 0 and r["best_lap"] < best:
            best = r["best_lap"]
        rate = count / max(1e-9, time.time() - t0) * 3600.0
        tag = (f"{r['best_lap']:.3f}s  [{DAGGER - r['best_lap']:+.2f} vs DAgger]"
               if r["laps"] > 0 else f"no lap ({r['max_dist']:.0f}m)")
        print(f"[trial {trial.number:4d}] {tag} | done={count} | {rate:.0f}/hr "
              f"| best={best if best < 1e9 else float('nan'):.3f}s", flush=True)

    kill_torcs()
    print(f"\n[Done] {count} trials | best {best:.3f}s", flush=True)


if __name__ == "__main__":
    main()

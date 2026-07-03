# WSL Optuna worker for the v4 sqrt-physics teacher (legacy).
import os
import sys
import math
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna

from config import Config
from search.optuna_teacher_v3 import make_storage
from search.optuna_teacher_linux import launch_torcs, kill_torcs, install_practice_xml, TORCS_HOME  # noqa: F401
from agents.teacher_controller_v4 import TeacherController, sample_params

DAGGER_BASELINE_LAP = 106.96
INF = float("inf")


def evaluate_v4(params, port=3001, n_laps=1, timeout_s=300.0):
    import core.snakeoil3_gym as snakeoil3

    controller = TeacherController(params)
    lap_times, max_dist = [], 0.0
    t0 = time.time()

    try:
        saved = sys.argv
        sys.argv = [sys.argv[0]]
        client = snakeoil3.Client(p=port)
        sys.argv = saved
    except Exception:
        return {"best_lap": INF, "avg_lap": INF, "laps": 0, "max_dist": 0.0}

    controller.reset()
    client.MAX_STEPS = int(timeout_s * 50)
    last_lap = None

    try:
        client.get_servers_input()
        while time.time() - t0 < timeout_s and len(lap_times) < n_laps:
            raw = client.S.d
            action = controller.act(raw)
            steer, ab = float(action[0]), float(action[1])
            accel, brake = (ab, 0.0) if ab >= 0 else (0.0, -ab)
            client.R.d["steer"] = steer
            client.R.d["accel"] = accel
            client.R.d["brake"] = brake
            client.R.d["gear"] = controller.get_gear()

            client.respond_to_server()
            client.get_servers_input()

            dist = float(raw.get("distRaced", 0.0))
            max_dist = max(max_dist, dist)
            llt = float(raw.get("lastLapTime", 0.0))
            if llt > 0 and llt != last_lap:
                last_lap = llt
                lap_times.append(llt)

            angle = float(raw.get("angle", 0.0))
            tp = float(raw.get("trackPos", 0.0))
            dmg = float(raw.get("damage", 0.0))
            if (math.cos(angle) < Config.torcs.backwards_cos_threshold
                    or dmg > Config.torcs.max_damage
                    or abs(tp) > Config.torcs.offtrack_trackpos_threshold):
                client.R.d["meta"] = True
                client.respond_to_server()
                break
    except Exception:
        pass

    try:
        client.R.d["meta"] = True
        client.respond_to_server()
    except Exception:
        pass

    return {
        "best_lap": min(lap_times) if lap_times else INF,
        "avg_lap": sum(lap_times) / len(lap_times) if lap_times else INF,
        "laps": len(lap_times),
        "max_dist": max_dist,
    }


def objective_value(result):
    if result["laps"] > 0 and result["best_lap"] < INF:
        return result["best_lap"]
    return 999.0 - result["max_dist"] / (Config.torcs.track_length_m / 100.0)


def n_completed(study):
    return len([t for t in study.trials
                if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])


def main():
    ap = argparse.ArgumentParser(description="WSL Optuna driver for the V4 physics teacher")
    ap.add_argument("--study-name", default="teacher_v4_ow1")
    ap.add_argument("--storage", default=None)
    ap.add_argument("--n-trials", type=int, default=1000000)
    ap.add_argument("--n-laps", type=int, default=1)
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--launch", choices=["r", "menu"], default="r")
    ap.add_argument("--display", default=":1")
    args = ap.parse_args()

    storage_url = args.storage or os.environ.get("OPTUNA_STORAGE") or "sqlite:///optuna_teacher_v4.db"
    backend = "postgres" if "postgres" in storage_url else ("sqlite" if "sqlite" in storage_url else "?")
    study = optuna.load_study(study_name=args.study_name, storage=make_storage(storage_url))
    race_xml = install_practice_xml(args.port)

    print(f"\n{'='*60}")
    print(f"  V4 physics driver  [storage={backend}] [launch={args.launch}]")
    print(f"  Study: {args.study_name} | completed: {n_completed(study)} | target: {args.n_trials}")
    print(f"  port={args.port} | n_laps={args.n_laps} | race={race_xml}")
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()
    count = 0
    best_seen = INF
    last_check = 0
    try:
        global_done = n_completed(study)
    except Exception:
        global_done = 0

    while global_done < args.n_trials:
        if count - last_check >= 25:
            try:
                global_done = n_completed(study)
            except Exception as e:
                print(f"[warn] count query failed: {e}", flush=True)
            last_check = count
            if global_done >= args.n_trials:
                break

        try:
            trial = study.ask()
        except Exception as e:
            print(f"[warn] ask failed ({e}); retry 5s", flush=True)
            time.sleep(5); continue

        params = sample_params(trial)

        launch_torcs(race_xml, args.port, args.launch, args.display)
        try:
            result = evaluate_v4(params, port=args.port, n_laps=args.n_laps, timeout_s=args.timeout_s)
        except Exception as e:
            print(f"[trial {trial.number}] eval error: {e}", flush=True)
            result = {"best_lap": INF, "avg_lap": INF, "laps": 0, "max_dist": 0.0}
        finally:
            kill_torcs()

        val = objective_value(result)
        try:
            study.tell(trial, val)
        except Exception:
            time.sleep(5)
            try:
                study.tell(trial, val)
            except Exception as e2:
                print(f"[warn] tell failed ({e2}); skip", flush=True)

        count += 1
        global_done += 1
        if result["laps"] > 0 and result["best_lap"] < best_seen:
            best_seen = result["best_lap"]

        rate = count / max(1e-9, time.time() - t0) * 3600.0
        if result["laps"] > 0:
            tag = f"{result['best_lap']:.3f}s  [{DAGGER_BASELINE_LAP - result['best_lap']:+.2f} vs DAgger]"
        else:
            tag = f"no lap ({result['max_dist']:.0f}m)"
        print(f"[trial {trial.number:4d}] {tag} | done={count} | {rate:.0f}/hr "
              f"| best={best_seen if best_seen < 1e9 else float('nan'):.3f}s", flush=True)

    kill_torcs()
    print(f"\n[Done] {count} trials in {(time.time()-t0)/3600:.2f}h | best: {best_seen:.3f}s", flush=True)


if __name__ == "__main__":
    main()

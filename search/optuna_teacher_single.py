# Lean single-instance Optuna driver for teacher v3 (Windows, legacy).
# One TORCS on port 3001; per-trial evaluation over snakeoil UDP.

import os
import sys
import time
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna

from config import Config
from search.optuna_teacher_v3 import sample_params, make_storage, DAGGER_BASELINE_LAP
from agents.teacher_controller_v3 import evaluate_teacher
from training.multi_instance_torcs import setup_instances

HERE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TORCS_EXE  = Config.torcs.torcs_exe
INST0_DIR  = Config.multi.instance_dir_pattern.format(idx=0)   # D:\torcs\torcs_inst0
PORT       = Config.multi.base_port                            # 3001
STARTUP_S  = Config.multi.startup_wait_s                       # 12.0


def kill_torcs():
    os.system('taskkill /IM wtorcs.exe /F >nul 2>&1')
    os.system('taskkill /IM torcs.exe /F >nul 2>&1')
    time.sleep(1.0)


def launch_torcs():
    # Launch ONE TORCS from torcs_inst0 and start the race via autostart_win.py,
    # invoked with THIS interpreter (the pyenv python has pyautogui). This mirrors
    # the working pool launch. Doing the autostart here (once per launch) means
    # snakeoil connects on the first try and never fires its own broken self-heal
    # relaunch (snakeoil3_gym._relaunch_torcs_windows calls a bare `python
    # autostart_win.py` that resolves to a python WITHOUT pyautogui).
    #
    # We don't track the wtorcs process (it hands off to a child); we relaunch by
    # image name + reset via meta between trials, mirroring TorcsSACEnv.
    kill_torcs()
    env = dict(os.environ)
    env["TORCS_DATA"] = INST0_DIR
    subprocess.Popen([TORCS_EXE, f"-p{PORT}"], cwd=INST0_DIR, env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    ap = subprocess.Popen([sys.executable, os.path.join(HERE, "autostart_win.py")],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ap.wait(timeout=45)          # autostart needs ~14-16s before keystrokes
    except subprocess.TimeoutExpired:
        ap.kill()
    time.sleep(2.0)


def objective_value(result: dict) -> float:
    # Lower = better. Lap time if it lapped, else distance-scaled penalty (matches pool).
    if result["laps"] > 0 and result["best_lap"] < float("inf"):
        return result["best_lap"]
    return 999.0 - result["max_dist"] / (Config.torcs.track_length_m / 100.0)


def n_completed(study) -> int:
    return len([t for t in study.trials
                if t.state in (optuna.trial.TrialState.COMPLETE,
                               optuna.trial.TrialState.PRUNED)])


def main():
    ap = argparse.ArgumentParser(description="Lean single-instance Optuna driver for Teacher V3")
    ap.add_argument("--study-name", default="teacher_v3")
    ap.add_argument("--storage",    default=None,
                    help="DB URL; if omitted uses $OPTUNA_STORAGE env var, else local sqlite")
    ap.add_argument("--n-trials",   type=int, default=2000, help="Target TOTAL completed trials in the study")
    ap.add_argument("--n-laps",     type=int, default=2)
    ap.add_argument("--port",       type=int, default=3001)
    ap.add_argument("--timeout-s",  type=float, default=300.0, help="Max seconds per trial evaluation")
    ap.add_argument("--relaunch-every", type=int, default=30, help="Relaunch TORCS every N trials (memory leak)")
    args = ap.parse_args()

    storage_url = args.storage or os.environ.get("OPTUNA_STORAGE") or "sqlite:///optuna_teacher_v3.db"
    backend = "postgres" if "postgres" in storage_url else ("sqlite" if "sqlite" in storage_url else storage_url.split(":")[0])
    study = optuna.load_study(study_name=args.study_name, storage=make_storage(storage_url))

    print(f"\n{'='*60}")
    print(f"  Lean Optuna driver — single persistent TORCS  [storage={backend}]")
    print(f"  Study: {args.study_name} | already completed: {n_completed(study)} | target: {args.n_trials}")
    print(f"  n_laps={args.n_laps} | relaunch every {args.relaunch_every} trials")
    print(f"{'='*60}\n", flush=True)

    # Ensure torcs_inst0 exists and its headless practice.xml is injected (idempotent).
    setup_instances(n_instances=1, base_port=PORT, verbose=False)
    launch_torcs()

    t0           = time.time()
    count        = 0
    since_launch = 0
    best_seen    = float("inf")

    last_global_check = 0
    try:
        global_done = n_completed(study)
    except Exception:
        global_done = 0

    while global_done < args.n_trials:
        # Re-query the SHARED completion count only every ~25 trials — on cloud
        # Postgres, study.trials fetches ALL trials and is expensive with many workers.
        if count - last_global_check >= 25:
            try:
                global_done = n_completed(study)
            except Exception as e:
                print(f"[warn] global-count query failed: {e}", flush=True)
            last_global_check = count
            if global_done >= args.n_trials:
                break

        try:
            trial = study.ask()
        except Exception as e:
            print(f"[warn] study.ask failed ({e}); retry in 5s", flush=True)
            time.sleep(5); continue

        params = sample_params(trial)

        try:
            result = evaluate_teacher(params, port=args.port,
                                      n_laps=args.n_laps, timeout_s=args.timeout_s,
                                      verbose=False)
        except Exception as e:
            print(f"[trial {trial.number}] eval error: {e} — relaunching TORCS", flush=True)
            launch_torcs(); since_launch = 0
            result = {"best_lap": float("inf"), "avg_lap": float("inf"), "laps": 0, "max_dist": 0.0}

        val = objective_value(result)
        try:
            study.tell(trial, val)
        except Exception as e:
            print(f"[warn] study.tell failed ({e}); retry once in 5s", flush=True)
            time.sleep(5)
            try:
                study.tell(trial, val)
            except Exception as e2:
                print(f"[warn] study.tell retry failed ({e2}); skipping", flush=True)
        count        += 1
        since_launch += 1
        global_done  += 1

        if result["laps"] > 0 and result["best_lap"] < best_seen:
            best_seen = result["best_lap"]

        rate = count / max(1e-9, time.time() - t0) * 3600.0
        if result["laps"] > 0:
            tag = f"{result['best_lap']:.3f}s  [{DAGGER_BASELINE_LAP - result['best_lap']:+.2f} vs DAgger]"
        else:
            tag = f"no lap ({result['max_dist']:.0f}m)"
        print(f"[trial {trial.number:4d}] {tag} | done={count} | {rate:.0f}/hr | best={best_seen if best_seen<1e9 else float('nan'):.3f}s", flush=True)

        if since_launch >= args.relaunch_every:
            print("[relaunch] periodic TORCS relaunch (memory-leak workaround)", flush=True)
            launch_torcs(); since_launch = 0

    kill_torcs()
    print(f"\n[Done] {count} trials this session in {(time.time()-t0)/3600:.2f}h | best lap seen: {best_seen:.3f}s", flush=True)


if __name__ == "__main__":
    main()

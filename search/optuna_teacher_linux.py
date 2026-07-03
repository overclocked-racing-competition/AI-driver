# Linux/WSL headless Optuna driver for teacher v3 + shared launch plumbing
# (install_practice_xml / launch_torcs / kill_torcs) reused by all v4-v6 drivers.
# Launch modes: "r" = torcs -r <race.xml> (batch, no X server; relaunched per trial),
# "menu" = Xvfb + xte keystrokes (fallback). One worker per port / per ~/.torcs HOME.
# Usage: python3 optuna_teacher_linux.py --study-name <name> --port 3001 [--storage URL]

import os
import sys
import time
import shutil
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna

from config import Config
from search.optuna_teacher_v3 import sample_params, make_storage, DAGGER_BASELINE_LAP
from agents.teacher_controller_v3 import evaluate_teacher

HERE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TORCS_HOME = os.path.expanduser("~/.torcs")


# ─────────────────────────────────────────────────────────────────────
#  TORCS config injection (Corkscrew + scr_server, results-only)
# ─────────────────────────────────────────────────────────────────────
def install_practice_xml(port: int) -> str:
    # Copy our headless Corkscrew/scr_server practice.xml into ~/.torcs
    # (idempotent) and patch the scr_server port so multiple workers differ.
    # Returns the absolute path to the installed race config.
    src = os.path.join(HERE, "practice.xml")
    dst_dir = os.path.join(TORCS_HOME, "config", "raceman")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "practice.xml")
    shutil.copy2(src, dst)

    # SCR convention: a scr_server robot at driver index i opens UDP port 3001+i.
    # Set the driver index to match --port so parallel workers use DISTINCT ports.
    # Requires scr_server.xml to map EVERY index (0-9) to car1-ow1 (see setup) —
    # otherwise idx>0 would silently drive the GT car1-trb1.
    idx = port - 3001
    with open(dst, "r", encoding="utf-8") as fh:
        xml = fh.read()
    xml = xml.replace('name="idx" val="0"', f'name="idx" val="{idx}"')
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(xml)
    return dst


def _patch_scr_port(port: int) -> None:
    # Set <attnum name="port" val="..."/> in the scr_server robot config, if it
    # exists at the usual location. Best-effort; ignored if the file is absent.
    candidates = [
        os.path.join(TORCS_HOME, "drivers", "scr_server", "scr_server.xml"),
        "/usr/local/share/games/torcs/drivers/scr_server/scr_server.xml",
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                txt = fh.read()
            import re
            new = re.sub(r'(<attnum name="port"\s+val=")\d+(")',
                         rf'\g<1>{port}\g<2>', txt)
            if new != txt and os.access(path, os.W_OK):
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(new)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
#  Launch / kill (Linux)
# ─────────────────────────────────────────────────────────────────────
def kill_torcs():
    # Scope the kill to THIS worker's TORCS only: its cmdline carries its own
    # ~/.torcs path (-l <HOME>/.torcs), so parallel workers don't kill each other.
    os.system(f"pkill -f '{TORCS_HOME}' >/dev/null 2>&1")
    time.sleep(0.7)


def _start_xvfb(display: str):
    # Start an Xvfb on `display` if the menu launch mode needs an X server.
    os.environ["DISPLAY"] = display
    # Only spawn if nothing is already listening on that display lock.
    lock = f"/tmp/.X{display.lstrip(':')}-lock"
    if not os.path.exists(lock):
        subprocess.Popen(f"Xvfb {display} -screen 0 640x480x24 >/tmp/xvfb{display}.log 2>&1 &",
                         shell=True)
        time.sleep(2.0)


def _autostart_xte():
    # Send the gym_torcs menu keystrokes via xte (xautomation) to start the
    # practice race. Sequence may need tuning to the actual menu layout — watch
    # the first launch over VNC if the race does not start.
    seq = ["key Return", "usleep 500000",
           "key Return", "usleep 500000",
           "key Up",     "usleep 300000",
           "key Up",     "usleep 300000",
           "key Return", "usleep 500000",
           "key Return"]
    try:
        subprocess.run(["xte", *seq], timeout=15)
    except Exception as e:
        print(f"[warn] xte autostart failed: {e}", flush=True)


def launch_torcs(race_xml: str, port: int, mode: str, display: str):
    kill_torcs()
    if mode == "r":
        # Batch race, results-only → no X server needed. scr_server blocks on
        # UDP until the client connects, so the race clock starts when we do.
        subprocess.Popen(
            f"torcs -r {race_xml} -nofuel -nodamage -nolaptime "
            f">/tmp/torcs_{port}.log 2>&1 &", shell=True)
        time.sleep(2.5)
    else:  # menu
        _start_xvfb(display)
        subprocess.Popen(
            f"torcs -nofuel -nodamage -nolaptime >/tmp/torcs_{port}.log 2>&1 &",
            shell=True)
        time.sleep(3.0)
        _autostart_xte()
        time.sleep(2.0)


# ─────────────────────────────────────────────────────────────────────
#  Optuna glue (shared with the Windows driver)
# ─────────────────────────────────────────────────────────────────────
def objective_value(result: dict) -> float:
    if result["laps"] > 0 and result["best_lap"] < float("inf"):
        return result["best_lap"]
    return 999.0 - result["max_dist"] / (Config.torcs.track_length_m / 100.0)


def n_completed(study) -> int:
    return len([t for t in study.trials
                if t.state in (optuna.trial.TrialState.COMPLETE,
                               optuna.trial.TrialState.PRUNED)])


def main():
    ap = argparse.ArgumentParser(description="Linux/WSL headless Optuna driver for Teacher V3")
    ap.add_argument("--study-name", default="teacher_wp")
    ap.add_argument("--storage", default=None,
                    help="DB URL; if omitted uses $OPTUNA_STORAGE, else local sqlite")
    ap.add_argument("--n-trials", type=int, default=2000, help="Target TOTAL completed trials in the study")
    ap.add_argument("--n-laps", type=int, default=1)
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--launch", choices=["r", "menu"], default="r",
                    help="r = torcs -r (headless, no X); menu = Xvfb + xte keystrokes")
    ap.add_argument("--display", default=":1", help="X display for --launch menu")
    args = ap.parse_args()

    storage_url = args.storage or os.environ.get("OPTUNA_STORAGE") or "sqlite:///optuna_teacher_v3.db"
    backend = "postgres" if "postgres" in storage_url else (
        "sqlite" if "sqlite" in storage_url else storage_url.split(":")[0])
    study = optuna.load_study(study_name=args.study_name, storage=make_storage(storage_url))

    race_xml = install_practice_xml(args.port)

    print(f"\n{'='*60}")
    print(f"  Linux Optuna driver  [storage={backend}] [launch={args.launch}]")
    print(f"  Study: {args.study_name} | already completed: {n_completed(study)} | target: {args.n_trials}")
    print(f"  port={args.port} | n_laps={args.n_laps} | race={race_xml}")
    print(f"{'='*60}\n", flush=True)

    t0        = time.time()
    count     = 0
    best_seen = float("inf")

    last_global_check = 0
    try:
        global_done = n_completed(study)
    except Exception:
        global_done = 0

    while global_done < args.n_trials:
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

        # Fresh, deterministic race per trial (cheap on Linux with -r).
        launch_torcs(race_xml, args.port, args.launch, args.display)
        try:
            result = evaluate_teacher(params, port=args.port,
                                      n_laps=args.n_laps, timeout_s=args.timeout_s,
                                      verbose=False)
        except Exception as e:
            print(f"[trial {trial.number}] eval error: {e}", flush=True)
            result = {"best_lap": float("inf"), "avg_lap": float("inf"), "laps": 0, "max_dist": 0.0}
        finally:
            kill_torcs()

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

        count       += 1
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
    print(f"\n[Done] {count} trials this session in {(time.time()-t0)/3600:.2f}h "
          f"| best lap seen: {best_seen:.3f}s", flush=True)


if __name__ == "__main__":
    main()

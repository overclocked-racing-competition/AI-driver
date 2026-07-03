# Diagnostic script for lap timing logic validation

import os
import sys
import time
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, TeacherV3Params
from agents.teacher_controller_v3 import TeacherController, load_params

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RACE = os.path.expanduser("~/.torcs/config/raceman/practice.xml")


def make_watch_config():
    # Build a NORMAL-display copy of the race config so the 3D race renders
    # (the training config uses 'results only' = headless). Returns its path.
    src = RACE if os.path.exists(RACE) else os.path.join(HERE, "practice.xml")
    with open(src, "r", encoding="utf-8", errors="ignore") as fh:
        xml = fh.read()
    xml = xml.replace('name="display mode" val="results only"',
                      'name="display mode" val="normal"')
    xml = xml.replace('name="display results" val="no"',
                      'name="display results" val="yes"')
    dst = os.path.expanduser("~/.torcs/config/raceman/practice_watch.xml")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(xml)
    return dst


def launch(mode, display, watch=False):
    os.system("pkill -f torcs >/dev/null 2>&1")
    time.sleep(1.0)
    if watch:
        # Graphical race via WSLg ($DISPLAY set by WSLg, usually :0) — renders
        # the race in a window on the Windows desktop so you can watch it.
        watch_xml = make_watch_config()
        subprocess.Popen(f"torcs -r {watch_xml} -nofuel -nodamage "
                         f">/tmp/torcs_watch.log 2>&1 &", shell=True)
        time.sleep(4.0)
        return
    if mode == "r":
        subprocess.Popen(f"torcs -r {RACE} -nofuel -nodamage -nolaptime "
                         f">/tmp/torcs_diag.log 2>&1 &", shell=True)
        time.sleep(2.5)
    else:
        os.environ["DISPLAY"] = display
        if not os.path.exists(f"/tmp/.X{display.lstrip(':')}-lock"):
            subprocess.Popen(f"Xvfb {display} -screen 0 640x480x24 >/tmp/xvfb.log 2>&1 &",
                             shell=True)
            time.sleep(2.0)
        subprocess.Popen("torcs -nofuel -nodamage -nolaptime >/tmp/torcs_diag.log 2>&1 &",
                         shell=True)
        time.sleep(3.0)
        seq = ["key Return", "usleep 500000", "key Return", "usleep 500000",
               "key Up", "usleep 300000", "key Up", "usleep 300000",
               "key Return", "usleep 500000", "key Return"]
        subprocess.run(["xte", *seq])
        time.sleep(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None, help="JSON params file; default = TeacherV3Params()")
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=200.0)
    ap.add_argument("--launch", choices=["r", "menu"], default="r")
    ap.add_argument("--display", default=":1")
    ap.add_argument("--watch", action="store_true",
                    help="Render the race in a graphical window (WSLg) so you can watch it")
    ap.add_argument("--no-launch", action="store_true",
                    help="Do NOT launch TORCS; connect to a race you already started from the menu")
    args = ap.parse_args()

    if args.params and os.path.exists(args.params):
        params = load_params(args.params)
        print(f"[params] tuned: {args.params}")
    else:
        params = TeacherV3Params()
        print("[params] DEFAULT TeacherV3Params()")

    print(f"[track ] config track_length_m = {Config.torcs.track_length_m} m")
    if not args.no_launch:
        launch(args.launch, args.display, args.watch)

    import core.snakeoil3_gym as snakeoil3
    ctrl = TeacherController(params)
    ctrl.reset()

    saved = sys.argv
    sys.argv = [sys.argv[0]]
    client = snakeoil3.Client(p=args.port)
    sys.argv = saved
    client.MAX_STEPS = int(args.seconds * 50)

    client.get_servers_input()
    last_lap = None
    laps = []
    prev_dist = 0.0
    max_track_pos = 0.0
    off_track_steps = 0
    t0 = time.time()

    while time.time() - t0 < args.seconds and len(laps) < args.laps:
        o = client.S.d
        a = ctrl.act(o)
        steer, ab = float(a[0]), float(a[1])
        accel, brake = (ab, 0.0) if ab >= 0 else (0.0, -ab)
        client.R.d["steer"] = steer
        client.R.d["accel"] = accel
        client.R.d["brake"] = brake
        client.R.d["gear"] = ctrl.get_gear(o)
        client.respond_to_server()
        client.get_servers_input()

        o = client.S.d
        tp = abs(float(o.get("trackPos", 0.0)))
        max_track_pos = max(max_track_pos, tp)
        if tp > 1.0:
            off_track_steps += 1

        llt = float(o.get("lastLapTime", 0.0))
        clt = float(o.get("curLapTime", 0.0))
        dr = float(o.get("distRaced", 0.0))
        dfs = float(o.get("distFromStart", 0.0))
        spx = float(o.get("speedX", 0.0))

        if llt > 0 and llt != last_lap:
            last_lap = llt
            laps.append(llt)
            print(f"  LAP {len(laps)}: lastLapTime={llt:7.3f}s | "
                  f"distRaced={dr:7.0f}m | since-prev={dr - prev_dist:7.0f}m | "
                  f"curLapTime={clt:7.3f}s | distFromStart={dfs:6.0f}m | speedX={spx:5.1f}")
            prev_dist = dr

    try:
        client.R.d["meta"] = True
        client.respond_to_server()
    except Exception:
        pass
    if not args.watch and not args.no_launch:
        os.system("pkill -f torcs >/dev/null 2>&1")
    else:
        print("[watch] TORCS zostaje otwarte — zamknij okno lub `pkill -f torcs`, gdy skończysz.")

    print(f"\n[summary] laps={len(laps)} times={[round(x,3) for x in laps]}")
    print(f"[summary] max |trackPos|={max_track_pos:.2f} "
          f"(>1.0 = off track surface) | off-track steps={off_track_steps}")
    print(f"[hint   ] a real Corkscrew lap should show since-prev ~= "
          f"{Config.torcs.track_length_m:.0f} m; large off-track = corner cutting")


if __name__ == "__main__":
    main()

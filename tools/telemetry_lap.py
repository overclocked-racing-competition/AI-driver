# Drive a study's best params and print min/max speed per 100 m bin
# (localizes lap-time losses). Requires port 3001 free (stop workers first).
# Usage: python3 telemetry_lap.py --controller v6 --study-name <name> --storage <url> --laps 2
import os, sys, math, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from config import Config
from search.optuna_teacher_v3 import make_storage
from search.optuna_teacher_linux import launch_torcs, kill_torcs, install_practice_xml


def ctrl_module(name):
    import importlib
    return importlib.import_module(f"teacher_controller_{name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="v6", choices=["v2", "v4", "v5", "v6"])
    ap.add_argument("--study-name", default="teacher_v2_ow1")
    ap.add_argument("--storage", default=os.environ.get("OPTUNA_STORAGE"))
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=260.0)
    args = ap.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=make_storage(args.storage))
    comp = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None and t.value < 900.0]
    if not comp:
        print("no lapping trial in study"); return
    best = min(comp, key=lambda t: t.value)
    print(f"[telemetry] {args.controller}  best trial #{best.number} = {best.value:.3f}s"
          f"   (completed lapping trials: {len(comp)})")
    # Print best params + flag any pegged at a search-range edge (→ widen that range).
    dists = best.distributions
    print("  best params (‹PEGGED› = at range edge, worth widening):")
    for k, v in sorted(best.params.items()):
        tag = ""
        d = dists.get(k)
        if d is not None and hasattr(d, "low") and hasattr(d, "high") and d.high > d.low:
            frac = (v - d.low) / (d.high - d.low)
            if frac <= 0.03:
                tag = f"  ‹PEGGED low [{d.low:g},{d.high:g}]›"
            elif frac >= 0.97:
                tag = f"  ‹PEGGED high [{d.low:g},{d.high:g}]›"
        print(f"    {k:22s} {v:>10.4g}{tag}" if isinstance(v, (int, float)) else f"    {k:22s} {v}{tag}")

    m = ctrl_module(args.controller)
    ctrl = m.TeacherController(m.params_from_optuna(best.params))

    race = install_practice_xml(args.port)
    launch_torcs(race, args.port, "r", ":1")

    import core.snakeoil3_gym as snakeoil3
    saved = sys.argv; sys.argv = [sys.argv[0]]
    client = snakeoil3.Client(p=args.port); sys.argv = saved
    client.MAX_STEPS = int(args.seconds * 50)
    ctrl.reset()

    BIN = 100.0
    nbins = int(Config.torcs.track_length_m / BIN) + 2
    vmin = [1e9] * nbins
    vmax = [0.0] * nbins
    laps, last, t0 = [], None, time.time()
    crash_dfs = None
    try:
        client.get_servers_input()
        while time.time() - t0 < args.seconds and len(laps) < args.laps:
            raw = client.S.d
            a = ctrl.act(raw)
            steer, ab = float(a[0]), float(a[1])
            accel, brake = (ab, 0.0) if ab >= 0 else (0.0, -ab)
            client.R.d["steer"] = steer; client.R.d["accel"] = accel
            client.R.d["brake"] = brake; client.R.d["gear"] = ctrl.get_gear()
            client.respond_to_server(); client.get_servers_input()

            raw = client.S.d
            spd = float(raw.get("speedX", 0.0))
            dfs = float(raw.get("distFromStart", 0.0))
            b = int(dfs / BIN)
            if 0 <= b < nbins and len(laps) >= 1:   # record only flying laps (skip standing lap 1)
                vmin[b] = min(vmin[b], spd)
                vmax[b] = max(vmax[b], spd)
            llt = float(raw.get("lastLapTime", 0.0))
            if llt > 0 and llt != last:
                last = llt; laps.append(llt)
            tp = float(raw.get("trackPos", 0.0)); ang = float(raw.get("angle", 0.0))
            if abs(tp) > Config.torcs.offtrack_trackpos_threshold or math.cos(ang) < Config.torcs.backwards_cos_threshold:
                crash_dfs = float(raw.get("distFromStart", 0.0))
                break
    except Exception as e:
        print("err", e)
    try:
        client.R.d["meta"] = True; client.respond_to_server()
    except Exception:
        pass
    kill_torcs()

    print(f"[telemetry] laps={[round(x,2) for x in laps]}"
          + (f"  CRASHED at distFromStart={crash_dfs:.0f}m" if crash_dfs else ""))
    print(f"\n  dist(m)   minSpeed  maxSpeed   (flying lap; low minSpeed = slow corner)")
    print("  " + "-" * 46)
    for b in range(nbins):
        if vmax[b] > 0:
            lo = vmin[b] if vmin[b] < 1e9 else 0.0
            bar = "#" * int(lo / 8)
            print(f"  {b*int(BIN):5d}    {lo:6.1f}    {vmax[b]:6.1f}   {bar}")
    allmax = max((v for v in vmax if v > 0), default=0)
    print(f"\n  top speed seen: {allmax:.1f} km/h")


if __name__ == "__main__":
    main()

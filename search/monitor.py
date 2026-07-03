# Live terminal dashboard for WSL Optuna workers. Parses /tmp/worker_*.log
# (no DB access -> no contention with workers). Run: python3 monitor.py
import os
import re
import time
import glob
import argparse

LINE = re.compile(
    r'\[trial\s+(\d+)\]\s+(.*?)\s+\|\s+done=(\d+)\s+\|\s+(\d+)/hr\s+\|\s+best=([\d.]+|nan)')


def parse_last(path):
    try:
        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None
    for ln in reversed(lines):
        m = LINE.search(ln)
        if m:
            body = m.group(2)
            if "no lap" in body:
                dm = re.search(r'\(([\d.]+)m\)', body)
                res, val = "no lap", (dm.group(1) + "m" if dm else "?")
            else:
                lm = re.search(r'([\d.]+)s', body)
                res, val = "LAP", (lm.group(1) + "s" if lm else "?")
            return {"trial": int(m.group(1)), "result": res, "val": val,
                    "done": int(m.group(3)), "rate": int(m.group(4)), "best": m.group(5)}
    txt = "".join(lines[-3:]).lower()
    return {"status": "connecting" if "waiting" in txt or "connect" in txt else "starting"}


def widx(p):
    m = re.search(r'(\d+)', os.path.basename(p))
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/tmp/worker_*.log")
    ap.add_argument("--study", default="teacher_ow1")
    ap.add_argument("--interval", type=float, default=4.0)
    args = ap.parse_args()

    try:
        while True:
            files = sorted(glob.glob(args.logs), key=widx)
            rows = [(widx(p), parse_last(p)) for p in files]

            done_total = sum(d["done"] for _, d in rows if d and "done" in d)
            bests = [float(d["best"]) for _, d in rows
                     if d and d.get("best") not in (None, "nan")]
            gbest = f"{min(bests):.3f}s" if bests else "—"
            n_lap = sum(1 for _, d in rows if d and d.get("result") == "LAP")

            os.system("clear")
            print(f"  Study: {args.study}    {time.strftime('%H:%M:%S')}    (Ctrl-C to exit; workers keep running)")
            print(f"  GLOBAL: workers={len(rows)}  done(this session)={done_total}  "
                  f"best lap={gbest}  lapping now={n_lap}/{len(rows)}   target 80s  (DAgger 106.96s)")
            print()
            print(f"  {'W':>2} {'port':>5} {'trial':>6} {'result':>7} {'lap/dist':>10} {'done':>5} {'/hr':>5} {'sess.best':>10}")
            print("  " + "-" * 62)
            for k, d in rows:
                if d is None:
                    print(f"  {k:>2} {3001+k:>5}   (no log yet)")
                elif "status" in d:
                    print(f"  {k:>2} {3001+k:>5}   {d['status']}...")
                else:
                    b = "—" if d["best"] == "nan" else f"{float(d['best']):.2f}s"
                    print(f"  {k:>2} {3001+k:>5} {d['trial']:>6} {d['result']:>7} "
                          f"{d['val']:>10} {d['done']:>5} {d['rate']:>5} {b:>10}")
            print()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[monitor] bye (workers still running)")


if __name__ == "__main__":
    main()

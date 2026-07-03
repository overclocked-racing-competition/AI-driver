# Phase 2: mass-start race-config generator + installer.
# Builds a multi-driver Corkscrew race (our scr_server + N bots) and writes it to
# ~/.torcs/config/raceman/massstart.xml, mirroring optuna_teacher_linux.install_practice_xml.
# The scr_server driver index = port - 3001 (must match the UDP client port).

import os

TORCS_HOME = os.path.expanduser("~/.torcs")


def build_massstart_xml(port: int, n_opponents: int, bot_module: str = "inferno",
                        display: str = "results only") -> str:
    # display: "results only" (headless training) or "normal" (GUI smoke test)
    # Grid order = "drivers list": place half the bots ahead of us, half behind,
    # so our car starts mid-pack (opponents on all sides for avoidance training).
    our_idx = port - 3001
    bots = [(bot_module, i) for i in range(n_opponents)]
    grid = bots[: n_opponents // 2] + [("scr_server", our_idx)] + bots[n_opponents // 2:]
    n_drivers = len(grid)

    driver_sections = "\n".join(
        f'    <section name="{k + 1}">\n'
        f'      <attnum name="idx" val="{idx}"/>\n'
        f'      <attstr name="module" val="{mod}"/>\n'
        f'    </section>'
        for k, (mod, idx) in enumerate(grid)
    )

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE params SYSTEM "params.dtd">
<params name="Practice">
  <section name="Header">
    <attstr name="name" val="Practice"/>
    <attstr name="description" val="Mass start ({n_opponents} opponents, {bot_module})"/>
    <attnum name="priority" val="100"/>
  </section>
  <section name="Tracks">
    <attnum name="maximum number" val="1"/>
    <section name="1">
      <attstr name="name" val="corkscrew"/>
      <attstr name="category" val="road"/>
    </section>
  </section>
  <section name="Races">
    <section name="1"><attstr name="name" val="Practice"/></section>
  </section>
  <section name="Practice">
    <attnum name="laps" val="20"/>
    <attstr name="type" val="practice"/>
    <attstr name="starting order" val="drivers list"/>
    <attstr name="restart" val="yes"/>
    <attstr name="display mode" val="{display}"/>
    <attstr name="display results" val="no"/>
    <attnum name="distance" unit="km" val="0"/>
    <section name="Starting Grid">
      <attnum name="rows" val="{n_drivers}"/>
      <attnum name="distance to start" val="25"/>
      <attnum name="distance between columns" val="20"/>
      <attnum name="offset within a column" val="10"/>
      <attnum name="initial speed" unit="km/h" val="0"/>
      <attnum name="initial height" unit="m" val="0.2"/>
    </section>
  </section>
  <section name="Drivers">
    <attnum name="maximum number" val="{n_drivers}"/>
    <attstr name="focused module" val="scr_server"/>
    <attnum name="focused idx" val="{our_idx}"/>
{driver_sections}
  </section>
</params>
'''


def install_massstart_xml(port: int, n_opponents: int, bot_module: str = "inferno") -> str:
    # Write the grid config into the TORCS raceman dir and return its path.
    dst_dir = os.path.join(TORCS_HOME, "config", "raceman")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "massstart.xml")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(build_massstart_xml(port, n_opponents, bot_module))
    return dst


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate a mass-start race config for inspection")
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--opponents", type=int, default=4)
    ap.add_argument("--bot", default="inferno")
    ap.add_argument("--display", default="results only",
                    choices=["results only", "normal"],
                    help="'normal' renders the race (GUI smoke test)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "race_configs", "massstart.xml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(build_massstart_xml(a.port, a.opponents, a.bot, a.display))
    print(f"wrote {a.opponents}-opponent grid ({a.bot}, display={a.display}) -> {out}")

# Teacher v2 (legacy): continuous lookahead-PID speed model,
#   target_speed = base_corner_speed + speed_per_meter * lookahead_distance,
# proportional throttle/brake; ABS / launch / racing line as v1. Superseded by v5/v6.
# Interface: act(raw_obs) -> [steer, accel_brake]; reset().

import argparse
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List

PI = math.pi


@dataclass
class TeacherParams:
    # Tunable parameters for the v2 continuous-speed controller.

    # --- Steering ---
    steer_gain: float = 26.0
    centering_gain: float = 0.18
    steer_damp: float = 0.25
    racing_line_entry: float = -0.30   # move inside on corner approach
    racing_line_exit: float = 0.20     # move outside on exit
    entry_sensor_thresh: float = 60.0
    exit_sensor_thresh: float = 120.0

    # --- Continuous speed model (the v2 core) ---
    max_speed: float = 260.0           # top target speed (km/h)
    base_corner_speed: float = 45.0    # target speed when look-ahead is ~0 (tight corner)
    speed_per_meter: float = 1.25      # +km/h of target per meter of open road ahead
    accel_gain: float = 2.0            # throttle = accel_gain * speed_error/100 (clamped 0..1)
    brake_gain: float = 2.5            # brake   = brake_gain * speed_error/100 (clamped -1..0)
    trail_throttle_floor: float = 0.10 # min throttle kept when on-target (smooths corner exit)
    lookahead_start: int = 4           # sensor index window for look-ahead distance
    lookahead_end: int = 16

    # --- ABS ---
    abs_enabled: bool = True
    abs_slip_threshold: float = 3.0
    abs_min_speed: float = 20.0
    abs_release_fraction: float = 0.7

    # --- TCS ---
    tcs_enabled: bool = True
    tcs_slip_threshold: float = 6.0
    tcs_min_speed: float = 25.0
    tcs_min_accel_factor: float = 0.15

    # --- Launch control ---
    launch_steps: int = 12

    # --- Gears ---
    rpm_upshift: float = 8200.0
    rpm_downshift: float = 3500.0
    # Multiplier on the base min-speed-for-gear table. <1.0 lets the car shift UP earlier
    # (escaping the gear-2 trap); >1.0 keeps it in lower gears longer.
    gear_speed_scale: float = 0.85
    base_min_speed_for_gear: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 45.0, 80.0, 115.0, 155.0, 195.0]
    )


class TeacherController:
    # v2 stateful expert controller. reset() at episode start, act(raw_obs) each step.

    def __init__(self, params: TeacherParams = None):
        self.p = params if params is not None else TeacherParams()
        self._prev_steer = 0.0
        self._gear = 1
        self._step = 0

    def reset(self):
        self._prev_steer = 0.0
        self._gear = 1
        self._step = 0

    def act(self, raw_obs: dict) -> np.ndarray:
        p = self.p
        self._step += 1

        angle     = float(raw_obs.get("angle", 0.0))
        track_pos = float(raw_obs.get("trackPos", 0.0))
        speed_x   = float(raw_obs.get("speedX", 0.0))
        rpm       = float(raw_obs.get("rpm", 0.0))
        track     = raw_obs.get("track", [200.0] * 19)
        wsv       = raw_obs.get("wheelSpinVel", [0.0, 0.0, 0.0, 0.0])

        # ---- Look-ahead distance: how much open road is straight ahead ----
        ahead_slice = track[p.lookahead_start:p.lookahead_end]
        lookahead   = float(min(ahead_slice)) if ahead_slice else 200.0

        # ---- Dynamic racing line ----
        target_pos = 0.0
        if lookahead < p.entry_sensor_thresh:
            target_pos += p.racing_line_entry
        elif lookahead > p.exit_sensor_thresh:
            target_pos += p.racing_line_exit

        # ---- Steering ----
        raw_steer = (angle * p.steer_gain / PI) - (track_pos - target_pos) * p.centering_gain
        steer = p.steer_damp * self._prev_steer + (1.0 - p.steer_damp) * raw_steer
        steer = float(np.clip(steer, -1.0, 1.0))
        self._prev_steer = steer

        # ---- Continuous target speed ----
        target_speed = p.base_corner_speed + p.speed_per_meter * lookahead
        target_speed = float(np.clip(target_speed, p.base_corner_speed, p.max_speed))

        # ---- Proportional throttle / brake ----
        if speed_x < 8.0 and self._step <= max(self.p.launch_steps, 40):
            ab = 1.0  # hard launch
        else:
            err = target_speed - speed_x   # km/h
            if err >= 0:
                ab = float(np.clip(err * p.accel_gain / 100.0, 0.0, 1.0))
                ab = max(ab, p.trail_throttle_floor)  # keep a little throttle for smoothness
            else:
                ab = float(np.clip(err * p.brake_gain / 100.0, -1.0, 0.0))

        # ---- ABS ----
        if p.abs_enabled and ab < 0 and speed_x > p.abs_min_speed:
            front_spin = (wsv[0] + wsv[1]) / 2.0
            if front_spin < p.abs_slip_threshold:
                ab = ab * p.abs_release_fraction

        # ---- TCS ----
        if p.tcs_enabled and ab > 0 and speed_x > p.tcs_min_speed:
            slip = (wsv[2] + wsv[3]) - (wsv[0] + wsv[1])
            if slip > p.tcs_slip_threshold:
                factor = max(p.tcs_min_accel_factor, p.tcs_slip_threshold / slip)
                ab = ab * factor

        ab = float(np.clip(ab, -1.0, 1.0))

        # ---- Gear ----
        self._gear = self._compute_gear(rpm, speed_x)

        return np.array([steer, ab], dtype=np.float32)

    def get_gear(self) -> int:
        return self._gear

    def _compute_gear(self, rpm: float, speed_x: float) -> int:
        p = self.p
        g = self._gear

        if self._step <= p.launch_steps:
            return 1

        # Scaled min-speed table — lets Optuna unlock higher gears earlier.
        msg = [s * p.gear_speed_scale for s in p.base_min_speed_for_gear]

        if rpm > p.rpm_upshift and g < 6:
            next_g = g + 1
            if speed_x >= msg[next_g]:
                g = next_g
        elif rpm < p.rpm_downshift and g > 1:
            g = g - 1

        while g > 1 and speed_x < msg[g] * 0.8:
            g -= 1

        return g


# ============================================================
# Optuna search space for v2
# ============================================================

# Maps Optuna trial parameter names (short) -> TeacherParams field names (full).
# Names not listed here are assumed identical in both.
_OPTUNA_TO_FIELD = {
    "rl_entry": "racing_line_entry",
    "rl_exit": "racing_line_exit",
    "entry_thr": "entry_sensor_thresh",
    "exit_thr": "exit_sensor_thresh",
    "base_corner_spd": "base_corner_speed",
    "speed_per_m": "speed_per_meter",
    "trail_floor": "trail_throttle_floor",
    "abs_slip": "abs_slip_threshold",
    "abs_min_spd": "abs_min_speed",
    "abs_rel": "abs_release_fraction",
    "tcs_slip": "tcs_slip_threshold",
    "tcs_min_spd": "tcs_min_speed",
    "tcs_min_fac": "tcs_min_accel_factor",
    "rpm_up": "rpm_upshift",
    "rpm_down": "rpm_downshift",
    "gear_scale": "gear_speed_scale",
}


def params_from_optuna(d: dict) -> "TeacherParams":
    # Build TeacherParams from a dict of Optuna trial parameters (as saved by
    # tune_teacher.py --mode export). Maps short Optuna names to dataclass fields
    # and ignores any keys that aren't valid fields.
    valid = set(TeacherParams.__dataclass_fields__.keys())
    kwargs = {}
    for k, v in d.items():
        field = _OPTUNA_TO_FIELD.get(k, k)
        if field in valid:
            kwargs[field] = v
    return TeacherParams(**kwargs)


def sample_params(trial) -> TeacherParams:
    # v2 Optuna search space — centered on the continuous-speed model.
    return TeacherParams(
        # Steering
        steer_gain        = trial.suggest_float("steer_gain",      15.0, 38.0),
        centering_gain    = trial.suggest_float("centering_gain",   0.08, 0.40),
        steer_damp        = trial.suggest_float("steer_damp",       0.0,  0.55),
        racing_line_entry = trial.suggest_float("rl_entry",        -0.60, 0.0),
        racing_line_exit  = trial.suggest_float("rl_exit",          0.0,  0.60),
        entry_sensor_thresh = trial.suggest_float("entry_thr",     40.0, 100.0),
        exit_sensor_thresh  = trial.suggest_float("exit_thr",      80.0, 180.0),

        # Continuous speed model
        max_speed         = trial.suggest_float("max_speed",      180.0, 300.0),
        base_corner_speed = trial.suggest_float("base_corner_spd", 30.0,  70.0),
        speed_per_meter   = trial.suggest_float("speed_per_m",      0.6,   2.2),
        accel_gain        = trial.suggest_float("accel_gain",       1.0,   5.0),
        brake_gain        = trial.suggest_float("brake_gain",       1.0,   6.0),
        trail_throttle_floor = trial.suggest_float("trail_floor",   0.0,   0.25),

        # ABS
        abs_slip_threshold   = trial.suggest_float("abs_slip",      1.0,   8.0),
        abs_min_speed        = trial.suggest_float("abs_min_spd",  10.0,  40.0),
        abs_release_fraction = trial.suggest_float("abs_rel",       0.40,  0.90),

        # TCS
        tcs_slip_threshold   = trial.suggest_float("tcs_slip",      2.0,  12.0),
        tcs_min_speed        = trial.suggest_float("tcs_min_spd",  15.0,  40.0),
        tcs_min_accel_factor = trial.suggest_float("tcs_min_fac",   0.05,  0.30),

        # Launch
        launch_steps = trial.suggest_int("launch_steps", 5, 25),

        # Gears
        rpm_upshift     = trial.suggest_float("rpm_up",   7000.0, 9200.0),
        rpm_downshift   = trial.suggest_float("rpm_down", 2500.0, 4500.0),
        gear_speed_scale= trial.suggest_float("gear_scale", 0.55, 1.10),
    )


# ============================================================
# Standalone diagnostic
# ============================================================

def run_diagnostic(n_episodes: int = 3, params: TeacherParams = None, verbose: bool = True):
    from config import Config
    from core.torcs_env_sac import TorcsSACEnv

    teacher = TeacherController(params or TeacherParams())
    env = TorcsSACEnv(stage=1)
    max_steps = Config.torcs.max_steps_per_episode

    all_dists, all_speeds, lap_times = [], [], []
    print(f"\n{'='*55}\n  Teacher v2 Diagnostic — {n_episodes} episodes\n{'='*55}\n")

    for ep in range(n_episodes):
        obs, info = env.reset()
        raw_obs = info.get("raw_obs", {})
        teacher.reset()
        ep_dist = ep_speed = 0.0
        last_lap = None

        for step in range(max_steps):
            action = teacher.act(raw_obs)
            obs, reward, terminated, truncated, info = env.step(action)
            raw_obs = info.get("raw_obs", {})
            ep_dist  = max(ep_dist,  float(raw_obs.get("distRaced", 0.0)))
            ep_speed = max(ep_speed, float(raw_obs.get("speedX", 0.0)))
            llt = float(raw_obs.get("lastLapTime", 0.0))
            if llt > 0 and llt != last_lap:
                last_lap = llt
                lap_times.append(llt)
                print(f"  [Ep {ep+1}] LAP: {llt:.2f}s")
            if verbose and step > 0 and step % 500 == 0:
                print(f"  [Ep {ep+1}] step {step:5d} | dist {float(raw_obs.get('distRaced',0)):7.1f}m "
                      f"| speed {float(raw_obs.get('speedX',0)):5.1f} | gear {int(raw_obs.get('gear',0))}")
            if terminated or truncated:
                break

        all_dists.append(ep_dist); all_speeds.append(ep_speed)
        print(f"  [Ep {ep+1}] max dist {ep_dist:.1f}m | max speed {ep_speed:.1f} km/h")

    env.close()
    print(f"\n  Max dist: {max(all_dists):.0f}m | Max speed: {max(all_speeds):.0f} km/h | Laps: {len(lap_times)}")
    if lap_times:
        print(f"  Best lap: {min(lap_times):.2f}s")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teacher v2 diagnostic")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_diagnostic(n_episodes=args.episodes, verbose=not args.quiet)

# Teacher v1 (legacy): parametric expert with zone-based speed model,
# ABS, launch control, TCS, and dynamic racing line. Optuna-tunable.
# Interface: act(raw_obs) -> [steer, accel_brake]; reset(). Superseded by v5/v6.

import argparse
import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List

PI = math.pi


# ============================================================
# Default parameters — tuned by Optuna in tune_teacher.py
# ============================================================

@dataclass
class TeacherParams:
    # All tunable parameters for the teacher controller.
    # Pass these to TeacherController to configure driving behaviour.
    # All defaults produce a functional driver (~1:30–1:40 without Optuna tuning).

    # --- Steering ---
    steer_gain: float = 28.0          # angle-to-steer proportional gain
    centering_gain: float = 0.20      # trackPos correction toward centre
    steer_damp: float = 0.25          # low-pass on steer output (0=none, 1=frozen)
    # Dynamic racing line: shift target trackPos on corner entry/exit
    # Positive = move to inside (negative trackPos); negative = move to outside.
    # Activated by sensor thresholds below.
    racing_line_entry: float = -0.3   # shift inward on corner approach (track narrows)
    racing_line_exit: float = 0.2     # shift outward on corner exit (track opens)
    entry_sensor_thresh: float = 60.0 # forward sensor reading that triggers entry shift
    exit_sensor_thresh: float = 120.0 # forward sensor reading that triggers exit shift

    # --- Speed management (look-ahead thresholds, km/h targets) ---
    # Sensors [5:15] cover a ~±4° arc in front; min of that window drives decisions.
    straight_thresh: float = 160.0    # ahead_min > this → full throttle
    medium_thresh: float = 85.0       # ahead_min > this → moderate throttle
    corner_thresh: float = 42.0       # ahead_min > this → braking zone
    # below corner_thresh → hairpin logic

    straight_accel: float = 1.0       # throttle on straights
    medium_accel: float = 0.75        # throttle on gentle curves
    corner_min_speed: float = 38.0    # min target speed in corners (km/h)
    corner_max_speed: float = 110.0   # max target speed in corners (km/h)
    hairpin_min_speed: float = 32.0   # min target speed in hairpins (km/h)
    hairpin_max_speed: float = 62.0   # max target speed in hairpins (km/h)
    corner_throttle: float = 0.30     # light throttle through corner
    hairpin_throttle: float = 0.18    # light throttle through hairpin
    corner_brake_cap: float = -0.85   # max braking in corners
    hairpin_brake_cap: float = -1.0   # max braking in hairpins

    # --- ABS (Anti-lock Braking System) ---
    # Detects wheel lock-up (front wheels stop spinning under braking) and releases
    # brake pressure to restore rotation. Allows later, harder braking points.
    abs_enabled: bool = True
    abs_slip_threshold: float = 3.0   # front wheelSpin below this → wheel locked
    abs_min_speed: float = 20.0       # ABS only active above this speed (km/h)
    abs_release_fraction: float = 0.7 # reduce brake to this fraction when locking

    # --- Traction Control ---
    tcs_enabled: bool = True
    tcs_slip_threshold: float = 5.0   # rear-front spin diff to trigger (rad/s)
    tcs_min_speed: float = 25.0       # TCS active above this speed (km/h)
    tcs_min_accel_factor: float = 0.15

    # --- Launch control ---
    # Hold 1st gear (prevent upshift) for this many steps after race start.
    # Prevents premature upshift that kills acceleration from standstill.
    launch_steps: int = 15            # ~0.3s at 50 Hz; winner used 15

    # --- Gear shift RPM thresholds (derived from vehicle XML; Optuna refines) ---
    rpm_upshift: float = 8000.0
    rpm_downshift: float = 3500.0
    min_speed_for_gear: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 45.0, 80.0, 115.0, 155.0, 195.0]
    )

    # --- Sensor window ---
    # Indices into the 19-element track[] array for look-ahead detection
    lookahead_start: int = 5          # sensor index start for ahead_min calc
    lookahead_end: int = 15           # sensor index end (exclusive)


# ============================================================
# Controller
# ============================================================

class TeacherController:
    # Stateful expert controller. Call reset() at the start of each episode.
    # Call act(raw_obs) each step to get [steer, accel_brake] in [-1, 1].

    def __init__(self, params: TeacherParams = None):
        self.p = params if params is not None else TeacherParams()
        self._prev_steer = 0.0
        self._gear = 1
        self._step = 0

    def reset(self):
        # Reset internal state at episode start.
        self._prev_steer = 0.0
        self._gear = 1
        self._step = 0

    def act(self, raw_obs: dict) -> np.ndarray:
        # Compute [steer, accel_brake] action from raw TORCS telemetry.
        #
        # Parameters
        # ----------
        # raw_obs : dict
        # Raw TORCS telemetry (same format as TorcsSACEnv info['raw_obs']).
        #
        # Returns
        # -------
        # np.ndarray shape (2,) — [steer, accel_brake], both in [-1, 1].
        p = self.p
        self._step += 1

        angle     = float(raw_obs.get("angle", 0.0))
        track_pos = float(raw_obs.get("trackPos", 0.0))
        speed_x   = float(raw_obs.get("speedX", 0.0))
        rpm       = float(raw_obs.get("rpm", 0.0))
        track     = raw_obs.get("track", [200.0] * 19)
        wsv       = raw_obs.get("wheelSpinVel", [0.0, 0.0, 0.0, 0.0])

        # ---- Look-ahead (which throttle/brake zone are we in?) ----
        ahead_slice = track[p.lookahead_start:p.lookahead_end]
        ahead_min   = float(min(ahead_slice)) if ahead_slice else 200.0

        # ---- Dynamic racing line ----
        # Adjust target centre position based on upcoming corner geometry.
        target_pos = 0.0
        if ahead_min < p.entry_sensor_thresh:
            target_pos += p.racing_line_entry   # inside on entry
        elif ahead_min > p.exit_sensor_thresh:
            target_pos += p.racing_line_exit    # outside on exit

        # ---- Steering ----
        raw_steer = (angle * p.steer_gain / PI) - (track_pos - target_pos) * p.centering_gain
        steer = p.steer_damp * self._prev_steer + (1.0 - p.steer_damp) * raw_steer
        steer = float(np.clip(steer, -1.0, 1.0))
        self._prev_steer = steer

        # ---- Throttle / Brake ----
        if speed_x < 8.0 and self._step <= 60:
            # Hard launch: full throttle off the line
            ab = 1.0
        elif ahead_min > p.straight_thresh:
            # Wide open road → full throttle
            ab = p.straight_accel
        elif ahead_min > p.medium_thresh:
            # Gentle curve → moderate throttle
            ab = p.medium_accel
        elif ahead_min > p.corner_thresh:
            # Corner approaching: brake to target speed
            t_min  = p.corner_min_speed
            t_max  = p.corner_max_speed
            frac   = (ahead_min - p.corner_thresh) / max(1.0, p.medium_thresh - p.corner_thresh)
            target = t_min + frac * (t_max - t_min)
            if speed_x > target:
                overshoot = (speed_x - target) / max(target, 1.0)
                ab = float(np.clip(-overshoot, p.corner_brake_cap, 0.0))
            else:
                ab = p.corner_throttle
        else:
            # Hairpin: firm braking to low target speed
            frac   = ahead_min / max(1.0, p.corner_thresh)
            target = p.hairpin_min_speed + frac * (p.hairpin_max_speed - p.hairpin_min_speed)
            if speed_x > target:
                overshoot = (speed_x - target) / max(target, 1.0)
                ab = float(np.clip(-overshoot * 1.3, p.hairpin_brake_cap, 0.0))
            else:
                ab = p.hairpin_throttle

        # ---- ABS ----
        # If we are braking and the front wheels have locked (stopped spinning relative
        # to vehicle speed), release brake pressure to restore wheel rotation.
        if p.abs_enabled and ab < 0 and speed_x > p.abs_min_speed:
            front_spin = (wsv[0] + wsv[1]) / 2.0
            # Front wheels should spin at roughly speed_x (km/h → rad/s proxy).
            # If they've almost stopped under heavy braking → wheel lock.
            if front_spin < p.abs_slip_threshold:
                # Release brake partially to restore rolling
                ab = ab * p.abs_release_fraction

        # ---- TCS ----
        if p.tcs_enabled and ab > 0 and speed_x > p.tcs_min_speed:
            slip = (wsv[2] + wsv[3]) - (wsv[0] + wsv[1])
            if slip > p.tcs_slip_threshold:
                factor = max(
                    p.tcs_min_accel_factor,
                    p.tcs_slip_threshold / slip,
                )
                ab = ab * factor

        ab = float(np.clip(ab, -1.0, 1.0))

        # ---- Gear ----
        self._gear = self._compute_gear(rpm, speed_x, ab)

        return np.array([steer, ab], dtype=np.float32)

    def get_gear(self) -> int:
        # Return the current gear for sending to TORCS (use in standalone mode).
        return self._gear

    def _compute_gear(self, rpm: float, speed_x: float, ab: float) -> int:
        # RPM-based automatic transmission with launch control and speed guard.
        p = self.p
        g = self._gear

        # Launch control: hold gear 1 for the first launch_steps
        if self._step <= p.launch_steps:
            return 1

        # Upshift
        if rpm > p.rpm_upshift and g < 6:
            next_g = g + 1
            if speed_x >= p.min_speed_for_gear[next_g]:
                g = next_g

        # Downshift
        elif rpm < p.rpm_downshift and g > 1:
            g = g - 1

        # Emergency downshift if speed fell below gear minimum
        while g > 1 and speed_x < p.min_speed_for_gear[g] * 0.8:
            g -= 1

        return g


# Maps v1 Optuna trial parameter names (short) -> TeacherParams field names (full).
_OPTUNA_TO_FIELD = {
    "rl_entry": "racing_line_entry",
    "rl_exit": "racing_line_exit",
    "entry_thr": "entry_sensor_thresh",
    "exit_thr": "exit_sensor_thresh",
    "straight_thr": "straight_thresh",
    "medium_thr": "medium_thresh",
    "corner_thr": "corner_thresh",
    "corner_min_spd": "corner_min_speed",
    "corner_max_spd": "corner_max_speed",
    "hairpin_min_spd": "hairpin_min_speed",
    "hairpin_max_spd": "hairpin_max_speed",
    "corner_thr_val": "corner_throttle",
    "hairpin_thr_val": "hairpin_throttle",
    "corner_brk": "corner_brake_cap",
    "hairpin_brk": "hairpin_brake_cap",
    "abs_slip": "abs_slip_threshold",
    "abs_min_spd": "abs_min_speed",
    "abs_rel": "abs_release_fraction",
    "tcs_slip": "tcs_slip_threshold",
    "tcs_min_spd": "tcs_min_speed",
    "tcs_min_fac": "tcs_min_accel_factor",
    "rpm_up": "rpm_upshift",
    "rpm_down": "rpm_downshift",
}


def params_from_optuna(d: dict) -> "TeacherParams":
    # Build TeacherParams from a dict of Optuna trial parameters (v1 names).
    valid = set(TeacherParams.__dataclass_fields__.keys())
    kwargs = {}
    for k, v in d.items():
        field = _OPTUNA_TO_FIELD.get(k, k)
        if field in valid:
            kwargs[field] = v
    return TeacherParams(**kwargs)


# ============================================================
# Standalone diagnostic runner
# ============================================================

def run_diagnostic(n_episodes: int = 5, params: TeacherParams = None, verbose: bool = True) -> dict:
    # Run the teacher controller in TorcsSACEnv and report lap quality.
    # Useful to verify the teacher can lap before running Optuna tuning.
    from config import Config
    from core.torcs_env_sac import TorcsSACEnv

    teacher = TeacherController(params or TeacherParams())
    env = TorcsSACEnv(stage=1)
    max_steps = Config.torcs.max_steps_per_episode

    all_max_dists  = []
    all_max_speeds = []
    all_lap_times  = []

    print(f"\n{'='*55}")
    print(f"  Teacher Controller Diagnostic — {n_episodes} episodes")
    print(f"{'='*55}\n")

    for ep in range(n_episodes):
        obs, info = env.reset()
        raw_obs = info.get("raw_obs", {})
        teacher.reset()

        ep_max_dist  = 0.0
        ep_max_speed = 0.0
        last_lap_recorded = None

        for step in range(max_steps):
            action = teacher.act(raw_obs)

            # In standalone mode, feed gear back to TORCS directly.
            # (In training, TorcsSACEnv handles gear internally via auto-transmission.)
            obs, reward, terminated, truncated, info = env.step(action)
            raw_obs = info.get("raw_obs", {})

            dist  = float(raw_obs.get("distRaced", 0.0))
            speed = float(raw_obs.get("speedX", 0.0))
            ep_max_dist  = max(ep_max_dist, dist)
            ep_max_speed = max(ep_max_speed, speed)

            llt = float(raw_obs.get("lastLapTime", 0.0))
            if llt > 0 and llt != last_lap_recorded:
                last_lap_recorded = llt
                all_lap_times.append(llt)
                print(f"  [Ep {ep+1}] LAP COMPLETED: {llt:.2f}s")

            if verbose and step > 0 and step % 500 == 0:
                gear = int(raw_obs.get("gear", 0))
                rpm  = float(raw_obs.get("rpm", 0.0))
                print(f"  [Ep {ep+1}] step {step:5d} | dist {dist:7.1f}m | "
                      f"speed {speed:5.1f} km/h | gear {gear} | rpm {rpm:5.0f}")

            if terminated or truncated:
                break

        all_max_dists.append(ep_max_dist)
        all_max_speeds.append(ep_max_speed)
        end = "TERM" if terminated else "TRUNC"
        print(f"  [Ep {ep+1}] {end:5s} | max dist {ep_max_dist:7.1f}m | "
              f"max speed {ep_max_speed:5.1f} km/h")

    env.close()

    print(f"\n{'='*55}")
    print(f"  Teacher Diagnostic Summary")
    print(f"{'='*55}")
    print(f"  Max dist ever:  {max(all_max_dists):7.1f} m  (track = 3602 m)")
    print(f"  Mean max dist:  {np.mean(all_max_dists):7.1f} m")
    print(f"  Max speed ever: {max(all_max_speeds):5.1f} km/h")
    print(f"  Laps completed: {len(all_lap_times)}")
    if all_lap_times:
        print(f"  Best lap:       {min(all_lap_times):.2f}s")
        print(f"  Avg lap:        {np.mean(all_lap_times):.2f}s")
    print()

    if max(all_max_dists) < 500:
        print("  ⚠ Teacher didn't reach 500m — check params or TORCS launch.")
    elif not all_lap_times:
        print("  ⚠ No laps completed — teacher drives but doesn't lap yet.")
        print("    Run tune_teacher.py to find parameters that complete laps.")
    else:
        print(f"  ✓ Teacher completed laps. Proceed to tune_teacher.py for sub-1:20.")
    print(f"{'='*55}\n")

    return {
        "max_dist":  max(all_max_dists),
        "max_speed": max(all_max_speeds),
        "laps":      len(all_lap_times),
        "lap_times": all_lap_times,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teacher controller diagnostic")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_diagnostic(n_episodes=args.episodes, verbose=not args.quiet)

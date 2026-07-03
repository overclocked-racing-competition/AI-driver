# Teacher v3 (legacy): static per-position profile — 12 speed waypoints +
# 12 trackPos waypoints interpolated over lap distance, ABS, trail braking.
# ~35 Optuna-tunable floats. Plateaued (blind to live sensors); superseded by v5/v6.

from __future__ import annotations

import math
import time
import json
import os
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

from config import Config, TeacherV3Params, CHECKPOINT_DIR


# ─────────────────────────────────────────────────────────────────────
#  Track constants
# ─────────────────────────────────────────────────────────────────────

TRACK_LENGTH_M  = Config.torcs.track_length_m  # 3602.0
N_WAYPOINTS     = TeacherV3Params.N_WAYPOINTS   # 12

# Waypoint positions (fraction of track, then meters)
WP_FRACTIONS    = [i / N_WAYPOINTS for i in range(N_WAYPOINTS)]
WP_METERS       = [f * TRACK_LENGTH_M for f in WP_FRACTIONS]

# F1 bolid approximate mass/decel for physics-based braking
# Max deceleration under braking ≈ 4g (conservative estimate for TORCS physics)
MAX_DECEL_G         = 3.5    # g (effective deceleration under braking)
GRAVITY             = 9.81   # m/s²
TORCS_SPEED_UNIT    = 1.0 / 3.6  # km/h → m/s conversion


def kmh_to_ms(v_kmh: float) -> float:
    return v_kmh * TORCS_SPEED_UNIT

def ms_to_kmh(v_ms: float) -> float:
    return v_ms / TORCS_SPEED_UNIT


# ─────────────────────────────────────────────────────────────────────
#  Interpolation utilities
# ─────────────────────────────────────────────────────────────────────

def interp_circular(dist_from_start: float, positions_m: List[float],
                     values: List[float], track_length: float) -> float:
    # Smooth circular interpolation around the track.
    #
    # dist_from_start is in meters. positions_m are the waypoint positions.
    # Handles wrap-around at the start/finish line.
    #
    # Returns: linearly interpolated value at dist_from_start.
    d = dist_from_start % track_length
    n = len(positions_m)

    # Find the two surrounding waypoints
    for i in range(n):
        next_i = (i + 1) % n
        p0 = positions_m[i]
        p1 = positions_m[next_i] if next_i != 0 else track_length

        # Handle wrap-around: last segment spans [last_wp → track_length → 0 → first_wp]
        if next_i == 0:
            p1 = track_length + positions_m[0]
            if d >= p0 or d < positions_m[0]:
                d_local = d if d >= p0 else d + track_length
                alpha = (d_local - p0) / max(p1 - p0, 1e-6)
                alpha = max(0.0, min(1.0, alpha))
                return values[i] * (1 - alpha) + values[next_i] * alpha

        if p0 <= d < p1:
            alpha = (d - p0) / max(p1 - p0, 1e-6)
            return values[i] * (1 - alpha) + values[next_i] * alpha

    return values[0]


def get_target_speed(dist_from_start: float, params: TeacherV3Params) -> float:
    # Target speed (km/h) at current track position, from waypoints.
    return interp_circular(
        dist_from_start, WP_METERS, params.speed_waypoints, TRACK_LENGTH_M
    )

def get_target_trackpos(dist_from_start: float, params: TeacherV3Params) -> float:
    # Target trackPos (−1..+1) at current track position, from waypoints.
    return interp_circular(
        dist_from_start, WP_METERS, params.trackpos_waypoints, TRACK_LENGTH_M
    )


# ─────────────────────────────────────────────────────────────────────
#  Physics-based braking distance
# ─────────────────────────────────────────────────────────────────────

def braking_distance_m(current_speed_kmh: float, target_speed_kmh: float,
                        decel_g: float = MAX_DECEL_G) -> float:
    # Minimum braking distance from current speed to target speed.
    #
    # Uses kinematics: v² = u² + 2·a·s → s = (v² − u²) / (2·a)
    # where a is the deceleration in m/s².
    # Returns 0 if current speed ≤ target speed.
    v_ms    = kmh_to_ms(current_speed_kmh)
    vt_ms   = kmh_to_ms(target_speed_kmh)
    if v_ms <= vt_ms:
        return 0.0
    decel   = decel_g * GRAVITY
    return max(0.0, (v_ms ** 2 - vt_ms ** 2) / (2.0 * decel))


# ─────────────────────────────────────────────────────────────────────
#  Forward track clearance (from the 19 rangefinder sensors)
# ─────────────────────────────────────────────────────────────────────

def forward_clearance(track_sensors: List[float],
                       start: int = 6, end: int = 13) -> float:
    # Minimum track edge distance in the forward-facing sensor cone.
    # track[9] is directly ahead; track[6:13] covers ±36° forward arc.
    # Clipped to 200m (sensor max).
    return float(min(track_sensors[start:end]))


def corner_severity(track_sensors: List[float],
                     start: int, end: int) -> float:
    # 0.0 = straight (all sensors > 200), 1.0 = tight hairpin.
    fc = forward_clearance(track_sensors, start, end)
    return max(0.0, 1.0 - fc / 200.0)


# ─────────────────────────────────────────────────────────────────────
#  ABS module
# ─────────────────────────────────────────────────────────────────────

class ABSController:
    # Anti-lock Braking System.
    #
    # Monitors wheel spin velocity relative to vehicle speed.
    # When wheel lockup is detected (spin drops below threshold),
    # reduces brake command to allow wheels to spin up again.
    #
    # wheel_spin_vel: list of 4 wheel angular velocities (rad/s from TORCS)
    # speed_x: longitudinal speed in km/h

    def __init__(self, slip_ratio_threshold: float = 0.25,
                 brake_cut: float = 0.35):
        self.slip_ratio_threshold = slip_ratio_threshold
        self.brake_cut = brake_cut
        self._abs_active = False
        self._brake_reduction = 0.0

    def update(self, wheel_spin_vel: List[float], speed_x_kmh: float,
               brake_cmd: float) -> float:
        # Return the ABS-modulated brake command.
        #
        # Lockup detection:
        # In TORCS, wheel spin velocity (rad/s) for a wheel with radius r
        # gives wheel peripheral speed = spin_vel × r.
        # We compare wheel speed to vehicle speed.
        # Typical wheel radius in TORCS F1: ~0.34m.
        # Locked wheel: spin approaches 0 while car moves.
        #
        # Simple heuristic (avoids need for exact wheel radius):
        # If speedX > 30 km/h and ANY wheel spin < 0.5×(average_spin / speed),
        # the wheel is approaching lockup.
        if speed_x_kmh < 10.0 or brake_cmd < 0.05:
            self._abs_active = False
            self._brake_reduction = 0.0
            return brake_cmd

        speed_ms = kmh_to_ms(speed_x_kmh)
        WHEEL_RADIUS = 0.34  # meters, approximate for TORCS F1

        # Convert wheel spin (rad/s) to linear speed (m/s)
        wheel_speeds = [abs(w) * WHEEL_RADIUS for w in wheel_spin_vel]

        # Slip ratio: (v_car - v_wheel) / v_car
        # 0 = no slip (rolling), 1 = full lockup
        slip_ratios = [max(0.0, (speed_ms - ws) / max(speed_ms, 0.1))
                       for ws in wheel_speeds]

        max_slip = max(slip_ratios)

        if max_slip > self.slip_ratio_threshold:
            # Wheel lockup detected — reduce brake
            self._abs_active = True
            self._brake_reduction = min(self.brake_cut, self._brake_reduction + 0.05)
        else:
            # Wheels rolling freely — can increase brake again
            self._abs_active = False
            self._brake_reduction = max(0.0, self._brake_reduction - 0.02)

        modulated_brake = brake_cmd * (1.0 - self._brake_reduction)
        return max(0.0, modulated_brake)

    @property
    def is_active(self) -> bool:
        return self._abs_active


# ─────────────────────────────────────────────────────────────────────
#  Gear manager
# ─────────────────────────────────────────────────────────────────────

class GearManager:
    # Automatic gear management matching the S3 driving_aids.py logic.
    # Ensures compatibility with the existing environment.

    UP_RPM_THRESH   = Config.aids.up_rpm_threshold    # 7000
    DOWN_RPM_THRESH = Config.aids.down_rpm_threshold   # 3000

    def __init__(self):
        self._gear = 1

    def update(self, rpm: float, speed_kmh: float, gear: int) -> int:
        # Return the recommended gear for current RPM.
        self._gear = max(1, int(gear))
        if rpm > self.UP_RPM_THRESH and self._gear < 6:
            self._gear += 1
        elif rpm < self.DOWN_RPM_THRESH and self._gear > 1:
            self._gear -= 1
        return self._gear

    def reset(self):
        self._gear = 1


# ─────────────────────────────────────────────────────────────────────
#  Main controller
# ─────────────────────────────────────────────────────────────────────

class TeacherController:
    # Full racing controller with ABS, racing line, and per-segment speed targets.
    #
    # act(raw_obs) → np.ndarray([steer, accel_brake])
    # steer:       −1.0 (full left) … +1.0 (full right)
    # accel_brake: −1.0 (full brake) … +1.0 (full throttle)
    #
    # This matches the S3 teacher interface so bc_pretrain.py and dagger.py
    # can use it without modification.

    def __init__(self, params: TeacherV3Params = None):
        self.params = params or TeacherV3Params()
        self.abs    = ABSController(
            slip_ratio_threshold=self.params.abs_slip_ratio,
            brake_cut=self.params.abs_brake_cut,
        )
        self.gear_mgr   = GearManager()
        self._step      = 0
        self._prev_steer = 0.0
        self._prev_dist  = 0.0

    def reset(self):
        self.abs         = ABSController(
            slip_ratio_threshold=self.params.abs_slip_ratio,
            brake_cut=self.params.abs_brake_cut,
        )
        self.gear_mgr.reset()
        self._step       = 0
        self._prev_steer = 0.0
        self._prev_dist  = 0.0

    # ── Raw sensor extraction ──────────────────────────────────────────

    @staticmethod
    def _get(raw_obs: dict, key: str, default: float = 0.0) -> float:
        v = raw_obs.get(key, default)
        if isinstance(v, (list, tuple)):
            return float(v[0]) if v else default
        return float(v) if v is not None else default

    @staticmethod
    def _get_list(raw_obs: dict, key: str, n: int,
                   default: float = 200.0) -> List[float]:
        v = raw_obs.get(key, [default] * n)
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v][:n] + [default] * max(0, n - len(v))
        return [default] * n

    # ── Steering ──────────────────────────────────────────────────────

    def _compute_steer(self, angle: float, track_pos: float,
                        dist_from_start: float) -> float:
        # Compute steering command.
        #
        # Combines:
        # 1. Angle correction: align car with track direction
        # 2. Racing-line correction: steer toward target trackPos
        # 3. Damping: smooth out steering changes
        p = self.params

        # Target trackPos from racing line
        target_tp = get_target_trackpos(dist_from_start, p)
        tp_error  = (track_pos - target_tp)   # positive = too far right → steer left

        # Angle correction (angle > 0 = car points left of track direction → steer right)
        angle_correction  = p.steer_angle_gain * angle / math.pi
        # TrackPos correction: move toward the racing line
        trackpos_correction = -p.steer_trackpos_gain * tp_error

        raw_steer = angle_correction + trackpos_correction

        # Damping: blend with previous steer
        damped_steer = (p.steer_damping * self._prev_steer
                        + (1.0 - p.steer_damping) * raw_steer)

        return float(np.clip(damped_steer, -p.steer_clip, p.steer_clip))

    # ── Throttle / Brake ──────────────────────────────────────────────

    def _compute_accel_brake(
        self, speed_x_kmh: float, dist_from_start: float,
        track_sensors: List[float], wheel_spin: List[float],
    ) -> float:
        # Compute the combined accel_brake output in [−1, +1].
        #
        # Positive = throttle, negative = brake.
        #
        # Strategy:
        # 1. Get target speed at current track position.
        # 2. Look AHEAD by braking_distance_m to find the lowest upcoming
        # target speed in the next segment. This is the "corner speed" we
        # must arrive at.
        # 3. If current speed > lookahead corner speed, apply brake.
        # 4. If wheels lock (ABS), modulate brake.
        # 5. If current speed < target and road is clear, apply throttle.
        p       = self.params
        fc      = forward_clearance(track_sensors,
                                     p.sensor_forward_start,
                                     p.sensor_forward_end)

        # -- Target speed at current position --
        target_now = get_target_speed(dist_from_start, p)

        # -- Target speed at next corner (lookahead) --
        # Compute required braking distance to slow from current to future speed.
        # Look ahead by an extra margin.
        lookahead_m = braking_distance_m(speed_x_kmh, target_now) + p.brake_lookahead_m

        # Sample target speed every 20m over the lookahead distance
        future_min_speed = target_now
        sample_step = 20.0
        samples = max(1, int(lookahead_m / sample_step))
        for s in range(1, samples + 1):
            future_dist = (dist_from_start + s * sample_step) % TRACK_LENGTH_M
            future_speed = get_target_speed(future_dist, p)
            future_min_speed = min(future_min_speed, future_speed)

        # -- Brake decision --
        # Brake when current speed exceeds the lookahead target significantly
        brake_trigger_speed = future_min_speed * p.target_speed_factor

        if (speed_x_kmh > brake_trigger_speed
                and speed_x_kmh > p.min_brake_speed_kmh):
            # How much over target are we?
            overspeed_frac = (speed_x_kmh - future_min_speed) / max(future_min_speed, 10.0)
            raw_brake = float(np.clip(p.brake_gain * overspeed_frac, 0.0, 1.0))

            # ABS modulation
            if p.abs_enabled:
                raw_brake = self.abs.update(wheel_spin, speed_x_kmh, raw_brake)

            return -raw_brake   # negative = brake in [−1, 0]

        # -- Throttle decision --
        # Use corner severity from sensors to cap throttle in corners
        sev = corner_severity(track_sensors, p.sensor_forward_start, p.sensor_forward_end)

        if fc >= p.sensor_brake_threshold:
            # Straight: full throttle
            throttle = p.throttle_max
        elif fc >= p.sensor_corner_fast:
            # Moderate corner: cap throttle
            throttle = p.accel_on_exit * (1.0 - sev * 0.3)
        elif fc >= p.sensor_corner_slow:
            # Tight corner: limited throttle
            throttle = p.throttle_in_corner * (1.0 - sev * 0.5)
        else:
            # Very tight: hairpin throttle (just enough to not stall)
            throttle = p.throttle_in_corner * 0.6

        return float(np.clip(throttle, 0.0, p.throttle_max))

    # ── Public API ────────────────────────────────────────────────────

    def act(self, raw_obs: dict) -> np.ndarray:
        # Main control loop. Returns [steer, accel_brake] in [−1, 1].
        # Compatible with the S3 teacher interface.
        p = self.params

        # -- Extract sensors --
        angle          = self._get(raw_obs, "angle", 0.0)
        track_pos      = self._get(raw_obs, "trackPos", 0.0)
        speed_x_kmh    = self._get(raw_obs, "speedX", 0.0)
        rpm            = self._get(raw_obs, "rpm", 0.0)
        gear           = self._get(raw_obs, "gear", 1.0)
        dist_from_start = self._get(raw_obs, "distFromStart", 0.0) % TRACK_LENGTH_M
        track_sensors  = self._get_list(raw_obs, "track", 19, default=200.0)
        wheel_spin     = self._get_list(raw_obs, "wheelSpinVel", 4, default=0.0)

        self._step += 1

        # -- Launch mode (standing start) --
        if self._step <= p.launch_steps:
            steer = self._compute_steer(angle, track_pos, dist_from_start)
            self._prev_steer = steer
            return np.array([steer, p.launch_throttle], dtype=np.float32)

        # -- Normal driving --
        steer       = self._compute_steer(angle, track_pos, dist_from_start)
        accel_brake = self._compute_accel_brake(
            speed_x_kmh, dist_from_start, track_sensors, wheel_spin
        )

        self._prev_steer = steer
        self._prev_dist  = dist_from_start

        return np.array([steer, accel_brake], dtype=np.float32)

    def get_gear(self, raw_obs: dict) -> int:
        # Auxiliary: recommended gear (for systems that need it separately).
        rpm      = self._get(raw_obs, "rpm", 0.0)
        speed    = self._get(raw_obs, "speedX", 0.0)
        gear     = self._get(raw_obs, "gear", 1.0)
        return self.gear_mgr.update(rpm, speed, gear)


# ─────────────────────────────────────────────────────────────────────
#  Serialization (for Optuna best-params export / import)
# ─────────────────────────────────────────────────────────────────────

def params_to_dict(params: TeacherV3Params) -> dict:
    # Serialize all scalar fields to a plain dict (for JSON export).
    return asdict(params)

def params_from_dict(d: dict) -> TeacherV3Params:
    # Deserialize a plain dict back to TeacherV3Params.
    # Filter to known fields only (forward-compatible)
    valid_fields = {f.name for f in TeacherV3Params.__dataclass_fields__.values()
                    if f.name != "N_WAYPOINTS"}
    filtered = {k: v for k, v in d.items() if k in valid_fields}
    return TeacherV3Params(**filtered)

def save_params(params: TeacherV3Params, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(params_to_dict(params), f, indent=2)
    print(f"[Teacher v3] Saved params → {path}")

def load_params(path: str) -> TeacherV3Params:
    with open(path) as f:
        d = json.load(f)
    params = params_from_dict(d)
    print(f"[Teacher v3] Loaded params from {path}")
    return params


# ─────────────────────────────────────────────────────────────────────
#  Standalone evaluation
# ─────────────────────────────────────────────────────────────────────

def evaluate_teacher(
    params: TeacherV3Params,
    port:   int   = 3001,
    n_laps: int   = 3,
    timeout_s: float = 480.0,
    verbose:   bool  = True,
) -> dict:
    # Run the teacher controller against a live TORCS instance.
    # Returns: {"best_lap": float, "avg_lap": float, "laps": int, "max_dist": float}
    #
    # Compatible with snakeoil3_gym.py UDP protocol.
    import core.snakeoil3_gym as snakeoil3

    controller = TeacherController(params)
    lap_times  = []
    max_dist   = 0.0
    t0         = time.time()

    if verbose:
        print(f"\n[TeacherV3] Evaluating on port {port}, target {n_laps} laps...")

    try:
        import sys
        saved_argv = sys.argv
        sys.argv   = [sys.argv[0]]
        client     = snakeoil3.Client(p=port)
        sys.argv   = saved_argv
    except Exception as e:
        print(f"[TeacherV3] Connection failed on port {port}: {e}")
        return {"best_lap": float("inf"), "avg_lap": float("inf"),
                "laps": 0, "max_dist": 0.0}

    controller.reset()
    client.MAX_STEPS = int(timeout_s * 50)  # 50 Hz
    last_lap  = None
    step      = 0

    try:
        client.get_servers_input()

        while time.time() - t0 < timeout_s and len(lap_times) < n_laps:
            raw_obs = client.S.d

            # Controller action
            action = controller.act(raw_obs)
            steer, accel_brake = float(action[0]), float(action[1])

            # Map combined accel_brake to separate accel / brake
            if accel_brake >= 0:
                accel, brake = accel_brake, 0.0
            else:
                accel, brake = 0.0, -accel_brake

            # Gear from controller
            gear = controller.get_gear(raw_obs)

            client.R.d["steer"] = steer
            client.R.d["accel"] = accel
            client.R.d["brake"] = brake
            client.R.d["gear"]  = gear

            client.respond_to_server()
            client.get_servers_input()

            step += 1
            dist = float(raw_obs.get("distRaced", 0.0))
            max_dist = max(max_dist, dist)

            # Lap detection
            llt = float(raw_obs.get("lastLapTime", 0.0))
            if llt > 0 and llt != last_lap:
                last_lap = llt
                lap_times.append(llt)
                if verbose:
                    print(f"  LAP {len(lap_times)}: {llt:.3f}s  (dist={dist:.0f}m)")

            # Episode reset detection
            angle     = float(raw_obs.get("angle", 0.0))
            track_pos = float(raw_obs.get("trackPos", 0.0))
            damage    = float(raw_obs.get("damage", 0.0))
            backwards = math.cos(angle) < Config.torcs.backwards_cos_threshold
            off_track = abs(track_pos) > Config.torcs.offtrack_trackpos_threshold
            too_dmgd  = damage > Config.torcs.max_damage

            if backwards or too_dmgd or off_track:
                if verbose:
                    reason = "backwards" if backwards else ("damage" if too_dmgd else "off-track")
                    print(f"  [Reset] {reason} at {dist:.0f}m")
                client.R.d["meta"] = True
                client.respond_to_server()
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[TeacherV3] Evaluation error on port {port}: {e}")

    # Always request a race restart so the NEXT evaluation starts from the grid.
    # The single-instance Optuna driver relies on this (keeps one TORCS alive and
    # resets via meta instead of relaunching the process every trial).
    try:
        client.R.d["meta"] = True
        client.respond_to_server()
    except Exception:
        pass

    if verbose:
        if lap_times:
            print(f"[TeacherV3] Laps: {len(lap_times)} | Best: {min(lap_times):.3f}s"
                  f" | Avg: {sum(lap_times)/len(lap_times):.3f}s")
        else:
            print(f"[TeacherV3] No laps completed | max_dist={max_dist:.0f}m")

    return {
        "best_lap": min(lap_times) if lap_times else float("inf"),
        "avg_lap":  sum(lap_times) / len(lap_times) if lap_times else float("inf"),
        "laps":     len(lap_times),
        "max_dist": max_dist,
    }


# ─────────────────────────────────────────────────────────────────────
#  CLI: quick test
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Teacher Controller V3 — standalone test")
    ap.add_argument("--port",   type=int, default=3001)
    ap.add_argument("--laps",   type=int, default=3)
    ap.add_argument("--params", type=str, default=None,
                    help="Path to JSON params file (from optuna_teacher_v3.py)")
    ap.add_argument("--save",   type=str, default=None,
                    help="Save default params to this JSON path")
    args = ap.parse_args()

    if args.save:
        default_params = TeacherV3Params()
        save_params(default_params, args.save)
        print(f"Default params saved to: {args.save}")
    else:
        params = load_params(args.params) if args.params else TeacherV3Params()
        results = evaluate_teacher(params, port=args.port,
                                   n_laps=args.laps, verbose=True)
        if results["laps"] > 0:
            print(f"\nBest lap: {results['best_lap']:.3f}s")
        else:
            print(f"\nDid not complete a lap. Max dist: {results['max_dist']:.0f}m")

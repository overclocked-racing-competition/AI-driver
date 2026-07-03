# Teacher v5: predictive late-braking speed model (kinematic braking envelope).
#   v_target = sqrt(v_apex^2 + brake_reach * (free_dist - margin)), clipped to max_speed
# Decouples straight-line speed from corner severity; strict generalization of v4
# (brake_reach=0 recovers it). Steering: aim-at-open-sensor. Near-redline rev-limit shift.
# Interface: act(raw_obs) -> [steer, accel_brake]; get_gear().
import math
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import List

PI = math.pi
SENSOR_ANGLES_DEG = [-45, -19, -12, -7, -4, -2.5, -1.7, -1, -0.5,
                     0, 0.5, 1, 1.7, 2.5, 4, 7, 12, 19, 45]
SENSOR_ANGLES_RAD = [math.radians(a) for a in SENSOR_ANGLES_DEG]


@dataclass
class TeacherParamsV5:
    # --- Predictive late-braking speed model ---
    max_speed: float = 310.0
    corner_coef: float = 15.0      # apex speed proxy: v_corner = corner_coef * sqrt(dist_i)
    brake_reach: float = 450.0     # predictive braking headroom [(km/h)^2 per m] — the late-braking term
    margin_m: float = 5.0
    min_speed: float = 40.0
    cone_lo: int = 3               # forward sensor window scanned for the binding corner
    cone_hi: int = 16
    # throttle/brake
    accel_gain: float = 3.4
    brake_gain: float = 5.0
    trail_throttle_floor: float = 0.06

    # --- Steering (aim-at-open racing line, from v4) ---
    k_aim: float = 0.5
    k_angle: float = 13.0
    k_center: float = 0.20
    steer_damp: float = 0.20
    aim_lo: int = 2
    aim_hi: int = 17

    # --- ABS / TCS / gears / launch ---
    abs_enabled: bool = True
    abs_slip_threshold: float = 3.0
    abs_min_speed: float = 20.0
    abs_release_fraction: float = 0.7
    tcs_enabled: bool = True
    tcs_slip_threshold: float = 6.0
    tcs_min_speed: float = 25.0
    tcs_min_accel_factor: float = 0.15
    launch_steps: int = 12
    rpm_upshift: float = 17800.0   # near the 18700 redline — keep the engine in its power band
    rpm_downshift: float = 9000.0
    n_gears: int = 7               # car1-ow1 has 7 forward gears


class TeacherController:
    def __init__(self, params: TeacherParamsV5 = None):
        self.p = params if params is not None else TeacherParamsV5()
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
        track     = list(raw_obs.get("track", [200.0] * 19))
        wsv       = raw_obs.get("wheelSpinVel", [0.0, 0.0, 0.0, 0.0])
        if len(track) < 19:
            track = track + [200.0] * (19 - len(track))

        # ---- Decoupled predictive speed ----
        # fwd = how far the track is open STRAIGHT ahead (narrow cone → ~200 m on a
        #       straight) → controls WHEN to brake. On a clear straight this is huge, so
        #       target clamps to max_speed regardless of corner tightness → full send.
        # sev = tightest reading over a wider forward arc → the upcoming corner APEX speed
        #       → controls HOW slow. Decoupling these is what lets straights use full
        #       speed while corners still slow down (the coupled formula capped both).
        fwd = max(track[7:12])
        lo = max(0, min(int(p.cone_lo), 18))
        hi = max(lo + 1, min(int(p.cone_hi), 19))
        sev = min(track[lo:hi])
        v_apex = p.corner_coef * math.sqrt(max(0.0, sev))
        vt = math.sqrt(v_apex * v_apex + p.brake_reach * max(0.0, fwd - p.margin_m))
        target_speed = float(np.clip(vt, p.min_speed, p.max_speed))

        # ---- Throttle / brake ----
        if speed_x < 8.0 and self._step <= max(p.launch_steps, 40):
            ab = 1.0
        else:
            err = target_speed - speed_x
            if err >= 0:
                ab = float(np.clip(err * p.accel_gain / 100.0, 0.0, 1.0))
                ab = max(ab, p.trail_throttle_floor)
            else:
                ab = float(np.clip(err * p.brake_gain / 100.0, -1.0, 0.0))

        if p.abs_enabled and ab < 0 and speed_x > p.abs_min_speed:
            front_spin = (wsv[0] + wsv[1]) / 2.0
            if front_spin < p.abs_slip_threshold:
                ab *= p.abs_release_fraction
        if p.tcs_enabled and ab > 0 and speed_x > p.tcs_min_speed:
            slip = (wsv[2] + wsv[3]) - (wsv[0] + wsv[1])
            if slip > p.tcs_slip_threshold:
                ab *= max(p.tcs_min_accel_factor, p.tcs_slip_threshold / slip)
        ab = float(np.clip(ab, -1.0, 1.0))

        # ---- Steering: aim at most-open direction + heading + centering ----
        alo = max(0, min(int(p.aim_lo), 18))
        ahi = max(alo + 1, min(int(p.aim_hi), 19))
        best_i = alo + int(np.argmax(track[alo:ahi]))
        aim = SENSOR_ANGLES_RAD[best_i]
        raw_steer = p.k_aim * aim + (angle * p.k_angle / PI) - track_pos * p.k_center
        steer = p.steer_damp * self._prev_steer + (1.0 - p.steer_damp) * raw_steer
        steer = float(np.clip(steer, -1.0, 1.0))
        self._prev_steer = steer

        self._gear = self._compute_gear(rpm, speed_x)
        return np.array([steer, ab], dtype=np.float32)

    def get_gear(self) -> int:
        return self._gear

    def _compute_gear(self, rpm: float, speed_x: float) -> int:
        p = self.p
        g = self._gear
        if self._step <= p.launch_steps:
            return 1
        # Shift near redline (short-shifting at ~8000 rpm capped top speed at
        # ~174 km/h). Gears 1-6 (SCR clamps to -1..6). Pure-RPM, no hunting.
        if rpm > p.rpm_upshift and g < 6:
            g += 1
        elif rpm < p.rpm_downshift and g > 1:
            g -= 1
        return g


def sample_params(trial) -> TeacherParamsV5:
    return TeacherParamsV5(
        max_speed        = trial.suggest_float("max_speed",       240.0, 330.0),
        corner_coef      = trial.suggest_float("corner_coef",       8.0,  24.0),
        brake_reach      = trial.suggest_float("brake_reach",      50.0, 1500.0),
        margin_m         = trial.suggest_float("margin_m",          0.0,  22.0),
        min_speed        = trial.suggest_float("min_speed",        28.0,  70.0),
        accel_gain       = trial.suggest_float("accel_gain",        1.5,   5.5),
        brake_gain       = trial.suggest_float("brake_gain",        2.5,   8.0),
        trail_throttle_floor = trial.suggest_float("trail_throttle_floor", 0.0, 0.20),
        k_aim            = trial.suggest_float("k_aim",             0.0,   1.2),
        k_angle          = trial.suggest_float("k_angle",           6.0,  28.0),
        k_center         = trial.suggest_float("k_center",          0.05,  0.50),
        steer_damp       = trial.suggest_float("steer_damp",        0.0,   0.50),
        abs_slip_threshold   = trial.suggest_float("abs_slip_threshold",   1.0, 8.0),
        abs_release_fraction = trial.suggest_float("abs_release_fraction", 0.40, 0.90),
        tcs_slip_threshold   = trial.suggest_float("tcs_slip_threshold",   2.0, 12.0),
        launch_steps     = trial.suggest_int("launch_steps",        5,    25),
        rpm_upshift      = trial.suggest_float("rpm_upshift",    15500.0, 18500.0),
        rpm_downshift    = trial.suggest_float("rpm_downshift",   6000.0, 12000.0),
    )


def params_from_optuna(d: dict) -> TeacherParamsV5:
    valid = set(TeacherParamsV5.__dataclass_fields__.keys())
    return TeacherParamsV5(**{k: v for k, v in d.items() if k in valid})

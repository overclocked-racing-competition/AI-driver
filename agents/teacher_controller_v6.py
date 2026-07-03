# Teacher v6: v5 decoupled speed model + parametric out-in-out racing line.
# Line target: turn = L/R sensor balance, sharp = forward-view reduction;
# entry -> -entry_gain*turn*sharp (outside), apex -> +apex_gain*turn*sharp (inside).
# Both gains searchable from 0, so the optimizer can fall back to v5 behavior.
# Interface: act(raw_obs) -> [steer, accel_brake]; get_gear(). Control only, no physics.
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

PI = math.pi
SENSOR_ANGLES_DEG = [-45, -19, -12, -7, -4, -2.5, -1.7, -1, -0.5,
                     0, 0.5, 1, 1.7, 2.5, 4, 7, 12, 19, 45]
SENSOR_ANGLES_RAD = [math.radians(a) for a in SENSOR_ANGLES_DEG]


@dataclass
class TeacherParamsV6:
    # --- Predictive decoupled speed (from v5) ---
    max_speed: float = 310.0
    corner_coef: float = 15.0
    brake_reach: float = 450.0
    margin_m: float = 5.0
    min_speed: float = 40.0
    cone_lo: int = 3
    cone_hi: int = 16
    accel_gain: float = 3.4
    brake_gain: float = 5.0
    trail_throttle_floor: float = 0.06

    # --- Steering: aim + out-in-out racing line ---
    k_aim: float = 0.35
    k_angle: float = 13.0
    k_line: float = 0.22       # pull toward the racing-line target trackPos
    steer_damp: float = 0.20
    entry_gain: float = 0.35   # how wide to set up on entry (outside)
    apex_gain: float = 0.45    # how tight to clip the apex (inside)
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
    rpm_upshift: float = 17800.0
    rpm_downshift: float = 9000.0
    n_gears: int = 7


class TeacherController:
    def __init__(self, params: Optional[TeacherParamsV6] = None):
        self.p = params if params is not None else TeacherParamsV6()
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

        # ---- Decoupled predictive speed (v5) ----
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
            if (wsv[0] + wsv[1]) / 2.0 < p.abs_slip_threshold:
                ab *= p.abs_release_fraction
        if p.tcs_enabled and ab > 0 and speed_x > p.tcs_min_speed:
            slip = (wsv[2] + wsv[3]) - (wsv[0] + wsv[1])
            if slip > p.tcs_slip_threshold:
                ab *= max(p.tcs_min_accel_factor, p.tcs_slip_threshold / slip)
        ab = float(np.clip(ab, -1.0, 1.0))

        # ---- Racing line: corner direction & sharpness ----
        left  = sum(track[10:16])   # angles +0.5 .. +12 (left side)
        right = sum(track[3:9])     # angles -12 .. -0.5 (right side)
        turn  = (left - right) / (left + right + 1.0)          # >0 => bends left
        sharp = min(1.0, max(0.0, 1.0 - fwd / 180.0))          # 0 straight .. 1 tight
        if sharp > 0.15 and abs(angle) < 0.12:
            target_pos = -p.entry_gain * turn * sharp          # set up WIDE (outside)
        else:
            target_pos = p.apex_gain * turn * sharp            # clip APEX (inside)
        target_pos = float(np.clip(target_pos, -0.9, 0.9))

        # ---- Steering: aim + heading + racing-line tracking ----
        alo = max(0, min(int(p.aim_lo), 18))
        ahi = max(alo + 1, min(int(p.aim_hi), 19))
        best_i = alo + int(np.argmax(track[alo:ahi]))
        aim = SENSOR_ANGLES_RAD[best_i]
        raw_steer = (p.k_aim * aim + angle * p.k_angle / PI
                     - p.k_line * (track_pos - target_pos))
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
        # gears 1-6: SCR clamps gear commands to -1..6 (a 7 would drop to neutral)
        if rpm > p.rpm_upshift and g < 6:
            g += 1
        elif rpm < p.rpm_downshift and g > 1:
            g -= 1
        return g


def sample_params(trial) -> TeacherParamsV6:
    return TeacherParamsV6(
        max_speed        = trial.suggest_float("max_speed",       240.0, 330.0),
        corner_coef      = trial.suggest_float("corner_coef",       8.0,  24.0),
        brake_reach      = trial.suggest_float("brake_reach",      50.0, 1500.0),
        margin_m         = trial.suggest_float("margin_m",          0.0,  22.0),
        min_speed        = trial.suggest_float("min_speed",        28.0,  70.0),
        accel_gain       = trial.suggest_float("accel_gain",        1.5,   5.5),
        brake_gain       = trial.suggest_float("brake_gain",        2.5,   8.0),
        trail_throttle_floor = trial.suggest_float("trail_throttle_floor", 0.0, 0.20),
        k_aim            = trial.suggest_float("k_aim",             0.0,   1.0),
        k_angle          = trial.suggest_float("k_angle",           6.0,  28.0),
        k_line           = trial.suggest_float("k_line",            0.05,  0.55),
        steer_damp       = trial.suggest_float("steer_damp",        0.0,   0.50),
        entry_gain       = trial.suggest_float("entry_gain",        0.0,   0.70),
        apex_gain        = trial.suggest_float("apex_gain",         0.0,   0.80),
        abs_slip_threshold   = trial.suggest_float("abs_slip_threshold",   1.0, 8.0),
        abs_release_fraction = trial.suggest_float("abs_release_fraction", 0.40, 0.90),
        tcs_slip_threshold   = trial.suggest_float("tcs_slip_threshold",   2.0, 12.0),
        launch_steps     = trial.suggest_int("launch_steps",        5,    25),
        rpm_upshift      = trial.suggest_float("rpm_upshift",    15500.0, 18500.0),
        rpm_downshift    = trial.suggest_float("rpm_downshift",   6000.0, 12000.0),
    )


def params_from_optuna(d: dict) -> TeacherParamsV6:
    valid = set(TeacherParamsV6.__dataclass_fields__.keys())
    return TeacherParamsV6(**{k: v for k, v in d.items() if k in valid})

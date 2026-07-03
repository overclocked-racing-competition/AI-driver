# Teacher v4 (legacy): grip-limit speed model, target = corner_coef * sqrt(reach)
# (v = sqrt(mu*g*R)); steering aims at the most-open sensor direction (k_aim).
# Throttle/ABS/TCS/gears inherited from v2. Coupled straights/corners; superseded by v5.
# Interface: act(raw_obs) -> [steer, accel_brake]; get_gear().
import math
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import List

PI = math.pi

# SCR track-sensor angles (deg) exactly as configured in snakeoil3_gym.py init.
SENSOR_ANGLES_DEG = [-45, -19, -12, -7, -4, -2.5, -1.7, -1, -0.5,
                     0, 0.5, 1, 1.7, 2.5, 4, 7, 12, 19, 45]
SENSOR_ANGLES_RAD = [math.radians(a) for a in SENSOR_ANGLES_DEG]


@dataclass
class TeacherParamsV4:
    # --- SQRT physics speed model ---
    max_speed: float = 305.0
    corner_coef: float = 18.0     # target = corner_coef * sqrt(reach - margin)  [km/h per sqrt(m)]
    margin_m: float = 5.0         # safety distance subtracted before a corner
    min_speed: float = 42.0
    cone_lo: int = 3              # wide forward sensor window (idx) → MIN reach anticipates corners
    cone_hi: int = 16
    # throttle/brake (from v2)
    accel_gain: float = 3.2
    brake_gain: float = 4.2
    trail_throttle_floor: float = 0.07

    # --- Steering: aim-at-most-open racing line ---
    k_aim: float = 0.5            # weight on steering toward the most open direction
    k_angle: float = 13.0         # heading alignment (applied as angle*k_angle/PI)
    k_center: float = 0.20        # centering on trackPos
    steer_damp: float = 0.20
    aim_lo: int = 2               # sensor window considered for the "most open" aim
    aim_hi: int = 17

    # --- ABS / TCS / gears / launch (from v2) ---
    abs_enabled: bool = True
    abs_slip_threshold: float = 3.0
    abs_min_speed: float = 20.0
    abs_release_fraction: float = 0.7
    tcs_enabled: bool = True
    tcs_slip_threshold: float = 6.0
    tcs_min_speed: float = 25.0
    tcs_min_accel_factor: float = 0.15
    launch_steps: int = 12
    rpm_upshift: float = 8300.0
    rpm_downshift: float = 3500.0
    gear_speed_scale: float = 0.80
    base_min_speed_for_gear: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 45.0, 80.0, 115.0, 155.0, 195.0])


class TeacherController:
    def __init__(self, params: TeacherParamsV4 = None):
        self.p = params if params is not None else TeacherParamsV4()
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

        # ---- SQRT physics target speed (MIN over a wide forward cone → anticipates corners) ----
        lo = max(0, min(int(p.cone_lo), 18))
        hi = max(lo + 1, min(int(p.cone_hi), 19))
        reach = min(track[lo:hi])
        target_speed = p.corner_coef * math.sqrt(max(0.0, reach - p.margin_m))
        target_speed = float(np.clip(target_speed, p.min_speed, p.max_speed))

        # ---- Throttle / brake (proportional to speed error) ----
        if speed_x < 8.0 and self._step <= max(p.launch_steps, 40):
            ab = 1.0  # launch
        else:
            err = target_speed - speed_x
            if err >= 0:
                ab = float(np.clip(err * p.accel_gain / 100.0, 0.0, 1.0))
                ab = max(ab, p.trail_throttle_floor)
            else:
                ab = float(np.clip(err * p.brake_gain / 100.0, -1.0, 0.0))

        # ---- ABS ----
        if p.abs_enabled and ab < 0 and speed_x > p.abs_min_speed:
            front_spin = (wsv[0] + wsv[1]) / 2.0
            if front_spin < p.abs_slip_threshold:
                ab *= p.abs_release_fraction
        # ---- TCS ----
        if p.tcs_enabled and ab > 0 and speed_x > p.tcs_min_speed:
            slip = (wsv[2] + wsv[3]) - (wsv[0] + wsv[1])
            if slip > p.tcs_slip_threshold:
                ab *= max(p.tcs_min_accel_factor, p.tcs_slip_threshold / slip)
        ab = float(np.clip(ab, -1.0, 1.0))

        # ---- Steering: aim at the most open direction + heading + centering ----
        alo = max(0, min(int(p.aim_lo), 18))
        ahi = max(alo + 1, min(int(p.aim_hi), 19))
        best_i = alo + int(np.argmax(track[alo:ahi]))
        aim = SENSOR_ANGLES_RAD[best_i]
        raw_steer = p.k_aim * aim + (angle * p.k_angle / PI) - track_pos * p.k_center
        steer = p.steer_damp * self._prev_steer + (1.0 - p.steer_damp) * raw_steer
        steer = float(np.clip(steer, -1.0, 1.0))
        self._prev_steer = steer

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
        msg = [s * p.gear_speed_scale for s in p.base_min_speed_for_gear]
        if rpm > p.rpm_upshift and g < 6:
            ng = g + 1
            if speed_x >= msg[ng]:
                g = ng
        elif rpm < p.rpm_downshift and g > 1:
            g -= 1
        while g > 1 and speed_x < msg[g] * 0.8:
            g -= 1
        return g


# ============================================================
# Optuna search space for v4 (field names used directly as param names)
# ============================================================
def sample_params(trial) -> TeacherParamsV4:
    return TeacherParamsV4(
        max_speed        = trial.suggest_float("max_speed",       230.0, 320.0),
        corner_coef      = trial.suggest_float("corner_coef",      11.0,  28.0),
        margin_m         = trial.suggest_float("margin_m",          0.0,  20.0),
        min_speed        = trial.suggest_float("min_speed",        30.0,  70.0),
        accel_gain       = trial.suggest_float("accel_gain",        1.5,   5.5),
        brake_gain       = trial.suggest_float("brake_gain",        2.0,   7.0),
        trail_throttle_floor = trial.suggest_float("trail_throttle_floor", 0.0, 0.20),
        k_aim            = trial.suggest_float("k_aim",             0.0,   1.2),
        k_angle          = trial.suggest_float("k_angle",           6.0,  28.0),
        k_center         = trial.suggest_float("k_center",          0.05,  0.50),
        steer_damp       = trial.suggest_float("steer_damp",        0.0,   0.50),
        abs_slip_threshold   = trial.suggest_float("abs_slip_threshold",   1.0, 8.0),
        abs_release_fraction = trial.suggest_float("abs_release_fraction", 0.40, 0.90),
        tcs_slip_threshold   = trial.suggest_float("tcs_slip_threshold",   2.0, 12.0),
        launch_steps     = trial.suggest_int("launch_steps",        5,    25),
        rpm_upshift      = trial.suggest_float("rpm_upshift",     7500.0, 9200.0),
        gear_speed_scale = trial.suggest_float("gear_speed_scale",  0.60,  1.05),
    )


def params_from_optuna(d: dict) -> TeacherParamsV4:
    valid = set(TeacherParamsV4.__dataclass_fields__.keys())
    return TeacherParamsV4(**{k: v for k, v in d.items() if k in valid})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="v4 controller quick sanity")
    ap.parse_args()
    c = TeacherController()
    for reach in (200, 100, 40, 15, 6):
        t = {"track": [reach] * 19, "angle": 0.0, "trackPos": 0.0, "speedX": 100.0, "rpm": 8000}
        a = c.act(t)
        print(f"reach={reach:4d}m -> steer={a[0]:+.2f} accel_brake={a[1]:+.2f}")

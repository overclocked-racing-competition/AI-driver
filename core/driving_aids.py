# Driving aids: shared action post-processing for training and submission
# Converts NN action [steer, accel_brake] into full TORCS command dict.
# Used identically by TorcsSACEnv and submit_agent.py.

from dataclasses import dataclass, field
from typing import List
import numpy as np

from config import Config


@dataclass
class AidsState:
    # Per-episode mutable state for driving aids
    prev_steer:   float = 0.0
    current_gear: int   = 1

    def reset(self) -> None:
        self.prev_steer   = 0.0
        self.current_gear = 1


def _auto_gear(raw_obs: dict, current_gear: int, action_cfg) -> int:
    # RPM-based automatic transmission, gears 1-6 (SCR clamps to -1..6)
    rpm = float(raw_obs.get("rpm", 0.0))
    g = current_gear if current_gear >= 1 else 1

    if rpm > action_cfg.rpm_upshift and g < action_cfg.max_gear:
        g += 1
    elif rpm < action_cfg.rpm_downshift and g > 1:
        g -= 1

    return g


def apply_aids(
    raw_obs: dict,
    nn_action: np.ndarray,
    state: AidsState,
    action_cfg=None,
) -> dict:
    # Convert 2-dim NN action into TORCS command dict {steer, accel, brake, gear}
    if action_cfg is None:
        action_cfg = Config.action

    steer       = float(nn_action[0])
    accel_brake = float(nn_action[1])

    # Steer rate limiter
    if action_cfg.steer_rate_limit_enabled:
        lim = action_cfg.steer_rate_limit
        steer = state.prev_steer + float(
            np.clip(steer - state.prev_steer, -lim, lim)
        )

    # Launch steer centering (low speed at start)
    if action_cfg.launch_assist_enabled and raw_obs:
        spd  = float(raw_obs.get("speedX",    0.0))
        clt  = float(raw_obs.get("curLapTime", 0.0))
        tpos = float(raw_obs.get("trackPos",   0.0))
        if spd < action_cfg.launch_release_speed and clt < action_cfg.launch_max_time:
            steer = float(np.clip(
                tpos * action_cfg.launch_centering_gain, -0.4, 0.4
            ))

    state.prev_steer = steer   # keep observation consistent with actual TORCS action

    # Accel/brake split + TCS
    if accel_brake >= 0:
        accel = accel_brake
        if action_cfg.tcs_enabled and raw_obs:
            speed_x = float(raw_obs.get("speedX", 0.0))
            if speed_x >= action_cfg.tcs_min_speed:
                wsv  = raw_obs.get("wheelSpinVel", [0.0, 0.0, 0.0, 0.0])
                slip = (wsv[2] + wsv[3]) - (wsv[0] + wsv[1])
                if slip > action_cfg.tcs_slip_threshold:
                    factor = max(
                        action_cfg.tcs_min_accel_factor,
                        action_cfg.tcs_slip_threshold / slip,
                    )
                    accel *= factor
        brake = 0.0
    else:
        accel = 0.0
        brake = abs(accel_brake)

    # Auto gear
    state.current_gear = _auto_gear(raw_obs, state.current_gear, action_cfg)

    return {
        "steer": steer,
        "accel": accel,
        "brake": brake,
        "gear":  state.current_gear,
    }

# Observation processing and normalization
# Transforms raw TORCS telemetry dicts into normalized numpy vectors for SAC.
# Stage 1: 32-dim (empty track), Stage 2: 68-dim (32 + 36 opponent sensors)

import numpy as np
from config import Config

_obs_cfg = Config.obs
_torcs_cfg = Config.torcs


def build_observation_stage1(raw_obs: dict, prev_steer: float = 0.0) -> np.ndarray:
    # Build 32-dim normalized obs vector from raw TORCS telemetry
    obs = np.zeros(_obs_cfg.stage1_dim, dtype=np.float32)

    # Track edge sensors (19 rangefinders, 0-200m)
    track = np.array(raw_obs.get("track", [0.0] * 19), dtype=np.float32)
    obs[0:19] = np.clip(track / _obs_cfg.track_sensor_max, -1.0, 1.0)

    # Speed X/Y
    obs[19] = np.clip(float(raw_obs.get("speedX", 0.0)) / _obs_cfg.speed_max, -1.0, 1.0)
    obs[20] = np.clip(float(raw_obs.get("speedY", 0.0)) / _obs_cfg.speed_max, -1.0, 1.0)

    # Angle (radians / pi)
    obs[21] = np.clip(float(raw_obs.get("angle", 0.0)) / np.pi, -1.0, 1.0)

    # Track position
    obs[22] = np.clip(float(raw_obs.get("trackPos", 0.0)), -1.0, 1.0)

    # RPM
    obs[23] = np.clip(float(raw_obs.get("rpm", 0.0)) / _obs_cfg.rpm_max, 0.0, 1.0)

    # Gear
    obs[24] = float(raw_obs.get("gear", 1)) / _obs_cfg.gear_max

    # Wheel spin velocities (4 wheels)
    wheel_spin = np.array(raw_obs.get("wheelSpinVel", [0.0] * 4), dtype=np.float32)
    obs[25:29] = np.clip(wheel_spin / _obs_cfg.wheel_spin_max, -1.0, 1.0)

    # Fractional lap progress
    obs[29] = np.clip(float(raw_obs.get("distFromStart", 0.0)) / _torcs_cfg.track_length_m, 0.0, 1.0)

    # Current lap time
    obs[30] = np.clip(float(raw_obs.get("curLapTime", 0.0)) / _obs_cfg.lap_time_max, 0.0, 1.0)

    # Previous steering command (keeps MDP Markov)
    obs[31] = np.clip(float(prev_steer), -1.0, 1.0)

    return obs


def build_observation_stage2(raw_obs: dict, prev_steer: float = 0.0) -> np.ndarray:
    # Build 68-dim obs vector (stage1 + opponent sensors)
    obs = np.zeros(_obs_cfg.stage2_dim, dtype=np.float32)

    # First 32 dims identical to stage 1
    obs[0:_obs_cfg.stage1_dim] = build_observation_stage1(raw_obs, prev_steer)

    # Opponent sensors (36 rangefinders)
    opponents = np.array(raw_obs.get("opponents", [200.0] * 36), dtype=np.float32)
    obs[_obs_cfg.stage1_dim:] = np.clip(opponents / _obs_cfg.opponent_sensor_max, 0.0, 1.0)

    return obs


def build_observation(raw_obs: dict, stage: int = 1, prev_steer: float = 0.0) -> np.ndarray:
    # Dispatcher: build obs for given curriculum stage
    if stage == 1:
        return build_observation_stage1(raw_obs, prev_steer)
    elif stage == 2:
        return build_observation_stage2(raw_obs, prev_steer)
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 1 or 2.")


def get_observation_dim(stage: int = 1) -> int:
    # Return observation dimensionality for the given stage
    if stage == 1:
        return _obs_cfg.stage1_dim
    elif stage == 2:
        return _obs_cfg.stage2_dim
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 1 or 2.")


def raw_obs_to_dict_safe(raw_obs: dict) -> dict:
    # Fill missing keys with safe defaults (guards against partial telemetry)
    defaults = {
        "angle": 0.0,
        "curLapTime": 0.0,
        "damage": 0.0,
        "distFromStart": 0.0,
        "distRaced": 0.0,
        "focus": [0.0] * 5,
        "fuel": 0.0,
        "gear": 1,
        "lastLapTime": 0.0,
        "opponents": [200.0] * 36,
        "racePos": 1,
        "rpm": 0.0,
        "speedX": 0.0,
        "speedY": 0.0,
        "speedZ": 0.0,
        "track": [0.0] * 19,
        "trackPos": 0.0,
        "wheelSpinVel": [0.0] * 4,
        "z": 0.0,
    }

    safe = defaults.copy()
    safe.update(raw_obs)
    return safe

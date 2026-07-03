# Modular reward shaping for SAC Racing AI
# Stage 1: time trial (empty track)
# Stage 2: multi-car race (opponent awareness, overtaking, collision)

import math
import numpy as np
from config import Config

_rw = Config.reward


def reward_stage1(
    raw_obs: dict,
    raw_obs_prev: dict,
    action: np.ndarray,
    action_prev: np.ndarray = None,
    off_track: bool = False,
    backwards: bool = False,
    lap_completed: bool = False,
    last_lap_time: float = 0.0,
) -> float:
    # Compute reward for Stage 1 (time trial, empty track)
    speed_x   = float(raw_obs.get("speedX", 0.0))
    angle     = float(raw_obs.get("angle", 0.0))
    track_pos = float(raw_obs.get("trackPos", 0.0))
    track     = np.array(raw_obs.get("track", [0.0] * 19), dtype=np.float32)
    wheel_spin = np.array(raw_obs.get("wheelSpinVel", [0.0] * 4), dtype=np.float32)

    # Progress (delta distRaced in meters)
    d_now  = float(raw_obs.get("distRaced", 0.0))
    d_prev = float(raw_obs_prev.get("distRaced", d_now)) if raw_obs_prev else d_now
    delta  = d_now - d_prev
    if delta < 0.0 or delta > 50.0:
        delta = 0.0
    r_progress = _rw.w_progress * delta

    # Per-step time cost
    r_time = -_rw.w_time

    # Track position penalty (quadratic)
    r_trackpos = _rw.w_trackpos * (track_pos ** 2)

    # Heading angle penalty
    r_angle = _rw.w_angle * abs(angle)

    # Steering smoothness penalty
    r_smooth = 0.0
    if action_prev is not None:
        steer_delta = abs(float(action[0]) - float(action_prev[0]))
        r_smooth = _rw.w_smoothness * steer_delta

    # Speed bonus on straights
    r_speed_bonus = 0.0
    forward_sensors = track[8:11]
    if len(forward_sensors) == 3 and np.min(forward_sensors) > _rw.straight_threshold:
        r_speed_bonus = _rw.w_speed_bonus * (speed_x / 300.0)

    # Corner speed reward
    r_cornering = 0.0
    forward_min = float(np.min(track[7:12])) if len(track) >= 12 else 200.0
    turn_sharpness = max(0.0, 1.0 - forward_min / _rw.straight_threshold)
    if turn_sharpness > 0.01:
        r_cornering = (
            _rw.w_cornering
            * (speed_x / 300.0)
            * abs(math.cos(angle))
            * max(0.0, 1.0 - abs(track_pos))
            * turn_sharpness
        )

    # Traction penalty (rear spin excess)
    rear_spin  = wheel_spin[2] + wheel_spin[3]
    front_spin = wheel_spin[0] + wheel_spin[1]
    spin_excess = max(0.0, rear_spin - front_spin - 5.0)
    r_traction = _rw.w_traction * spin_excess

    # Damage penalty
    r_damage = 0.0
    if raw_obs_prev is not None:
        cur_damage  = float(raw_obs.get("damage", 0.0))
        prev_damage = float(raw_obs_prev.get("damage", 0.0))
        delta = max(0.0, cur_damage - prev_damage)
        if delta > 0.0:
            r_damage = _rw.w_damage * min(delta, _rw.damage_cap)

    # Event-based rewards
    r_event = 0.0
    if off_track:
        r_event += _rw.penalty_offtrack
    if backwards:
        r_event += _rw.penalty_backwards
    if lap_completed and last_lap_time > 0.0:
        time_bonus = max(0.0, _rw.lap_target_time - last_lap_time)
        r_event += _rw.lap_bonus + time_bonus

    return (
        r_progress + r_time + r_trackpos + r_angle + r_smooth
        + r_speed_bonus + r_cornering + r_traction + r_damage + r_event
    )


def reward_stage2(
    raw_obs: dict,
    raw_obs_prev: dict,
    action: np.ndarray,
    action_prev: np.ndarray = None,
    off_track: bool = False,
    backwards: bool = False,
    lap_completed: bool = False,
    last_lap_time: float = 0.0,
) -> float:
    # Compute reward for Stage 2 (multi-car race)
    r_base = reward_stage1(
        raw_obs, raw_obs_prev, action, action_prev,
        off_track=off_track,
        backwards=backwards,
        lap_completed=lap_completed,
        last_lap_time=last_lap_time,
    )

    # Opponent proximity penalty
    r_opponent = 0.0
    opponents = np.array(raw_obs.get("opponents", [200.0] * 36), dtype=np.float32)
    min_opp_dist = opponents.min()
    if min_opp_dist < _rw.opponent_danger_dist:
        r_opponent = _rw.w_opponent_proximity * (1.0 - min_opp_dist / _rw.opponent_danger_dist)

    # Overtaking bonus
    r_overtake = 0.0
    if raw_obs_prev is not None:
        cur_pos  = int(raw_obs.get("racePos", 1))
        prev_pos = int(raw_obs_prev.get("racePos", 1))
        if cur_pos < prev_pos:
            r_overtake = _rw.bonus_overtake

    # Collision penalty
    r_collision = 0.0
    if raw_obs_prev is not None:
        cur_damage  = float(raw_obs.get("damage", 0.0))
        prev_damage = float(raw_obs_prev.get("damage", 0.0))
        if cur_damage > prev_damage:
            r_collision = _rw.penalty_collision

    return r_base + r_opponent + r_overtake + r_collision


def compute_reward(
    raw_obs: dict,
    raw_obs_prev: dict,
    action: np.ndarray,
    action_prev: np.ndarray = None,
    stage: int = 1,
    off_track: bool = False,
    backwards: bool = False,
    lap_completed: bool = False,
    last_lap_time: float = 0.0,
) -> float:
    # Dispatcher: compute reward for given curriculum stage
    kwargs = dict(
        raw_obs=raw_obs,
        raw_obs_prev=raw_obs_prev,
        action=action,
        action_prev=action_prev,
        off_track=off_track,
        backwards=backwards,
        lap_completed=lap_completed,
        last_lap_time=last_lap_time,
    )
    if stage == 1:
        return reward_stage1(**kwargs)
    elif stage == 2:
        return reward_stage2(**kwargs)
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 1 or 2.")

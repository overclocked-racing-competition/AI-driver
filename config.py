# S4-F Master Configuration

from __future__ import annotations
import os
import copy
import torch
from dataclasses import dataclass, field
from typing import List, Optional


# Paths
PROJECT_ROOT    = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR  = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR         = os.path.join(PROJECT_ROOT, "logs")
TELEMETRY_DIR   = os.path.join(PROJECT_ROOT, "telemetry")
TORCS_EXE       = r"D:\torcs\torcs\wtorcs.exe"
TORCS_CONFIG_DIR = r"D:\torcs\torcs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(TELEMETRY_DIR,  exist_ok=True)


# TORCS Environment
@dataclass
class TorcsConfig:
    host:                        str   = "localhost"
    track_name:                  str   = "corkscrew"
    track_category:              str   = "road"
    track_length_m:              float = 3602.0      # Laguna Seca Corkscrew
    vision:                      bool  = False
    base_port:                   int   = 3001
    port:                        int   = 3001
    tick_rate:                   int   = 50
    torcs_dir:                   str   = r"D:\torcs\torcs"
    torcs_exe:                   str   = r"D:\torcs\torcs\wtorcs.exe"
    max_steps_per_episode:       int   = 9000        # 3602m / (50Hz × avg speed) ≈ safe ceiling
    relaunch_every_n_episodes:   int   = 25
    sensor_angles:               str   = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
    offtrack_trackpos_threshold: float = 1.10        # |trackPos| > this → off-track
    max_damage:                  float = 5000.0
    backwards_cos_threshold:     float = 0.0         # cos(angle) < this → backwards
    grace_steps_offtrack:        int   = 30          # renamed in S4-F
    max_offtrack_steps:          int   = 30          # alias to match older environments
    progress_stuck_window:       int   = 250
    progress_stuck_min_dist:     float = 2.0
    reset_grace_steps:           int   = 15          # steps after reset before checking

torcs = TorcsConfig()


# Teacher V3 — per-segment racing-line controller
@dataclass
class TeacherV3Params:
    # Speed waypoints (12 x km/h, equidistant around 3602m track)
    speed_wp0:  float = 150.0   #    0m – Start / finish straight (conservative)
    speed_wp1:  float = 80.0    #  300m – Turn 2 Andretti Hairpin brake
    speed_wp2:  float = 100.0   #  600m – Turn 3
    speed_wp3:  float = 90.0    #  900m – Turn 4
    speed_wp4:  float = 140.0   # 1200m – Back straight
    speed_wp5:  float = 130.0   # 1500m – Back straight continuation (Turn 5/6)
    speed_wp6:  float = 100.0   # 1800m – Turn 6 approach
    speed_wp7:  float = 70.0    # 2100m – Corkscrew T8
    speed_wp8:  float = 60.0    # 2400m – Corkscrew T9 exit
    speed_wp9:  float = 50.0    # 2700m – Turn 10
    speed_wp10: float = 50.0    # 3000m – Turn 11 approach
    speed_wp11: float = 50.0    # 3300m – Turn 11 apex / exit

    # Racing line waypoints (trackPos: -1=left, 0=center, +1=right)
    tp_wp0:  float =  0.0
    tp_wp1:  float =  0.0
    tp_wp2:  float =  0.0
    tp_wp3:  float =  0.0
    tp_wp4:  float =  0.0
    tp_wp5:  float =  0.0
    tp_wp6:  float =  0.0
    tp_wp7:  float =  0.0
    tp_wp8:  float =  0.0
    tp_wp9:  float =  0.0
    tp_wp10: float =  0.0
    tp_wp11: float =  0.0

    # Steering
    steer_angle_gain:   float = 28.0   # how much to correct for car angle vs track (was 0.60)
    steer_trackpos_gain: float = 0.50  # how much to correct for lateral position
    steer_damping:       float = 0.25  # blend with previous steer (0=no damping)
    steer_clip:          float = 1.0   # max abs steering output

    # Braking
    brake_gain:          float = 1.8   # amplify braking command
    brake_lookahead_m:   float = 20.0  # extra lookahead distance for braking (m)
    target_speed_factor: float = 0.90  # start braking when speed > factor × target_speed
    min_brake_speed_kmh: float = 30.0  # never brake below this speed

    # ABS
    abs_enabled:         bool  = True
    abs_slip_ratio:      float = 0.25  # (v_wheel / v_car) drop threshold for lockup detect
    abs_brake_cut:       float = 0.35  # fraction to reduce brake by on ABS trigger

    # Throttle
    throttle_max:        float = 1.0   # full throttle cap
    throttle_in_corner:  float = 0.55  # max throttle allowed when cornering
    throttle_corner_threshold: float = 60.0  # track sensor min value below which we're in corner
    accel_on_exit:       float = 0.9   # throttle when past apex and accelerating out

    # Track sensor interpretation
    sensor_forward_start: int  = 6     # track sensor indices for forward clearance
    sensor_forward_end:   int  = 12    # track[6:12] = forward zone
    sensor_brake_threshold: float = 180.0  # min sensor value = straight (no braking)
    sensor_corner_fast:   float = 80.0    # above this = moderate corner
    sensor_corner_slow:   float = 40.0    # below this = tight corner

    # Launch sequence
    launch_throttle:     float = 0.95  # standing-start throttle
    launch_steps:        int   = 120   # steps to maintain launch mode (~2.4s at 50Hz)

    @property
    def speed_waypoints(self) -> List[float]:
        return [
            self.speed_wp0, self.speed_wp1, self.speed_wp2, self.speed_wp3,
            self.speed_wp4, self.speed_wp5, self.speed_wp6, self.speed_wp7,
            self.speed_wp8, self.speed_wp9, self.speed_wp10, self.speed_wp11,
        ]

    @property
    def trackpos_waypoints(self) -> List[float]:
        return [
            self.tp_wp0, self.tp_wp1, self.tp_wp2,  self.tp_wp3,
            self.tp_wp4, self.tp_wp5, self.tp_wp6,  self.tp_wp7,
            self.tp_wp8, self.tp_wp9, self.tp_wp10, self.tp_wp11,
        ]

    # Number of waypoints (keep in sync with the properties above)
    N_WAYPOINTS: int = 12


# Multi-Instance TORCS
@dataclass
class MultiInstanceConfig:
    # Parallel TORCS instances, each on a different UDP port
    n_instances:       int   = 6       # parallel TORCS instances (tune by RAM)
    base_port:         int   = 3001    # instance 0 = port 3001, 1 = 3002, etc.
    torcs_exe:         str   = TORCS_EXE
    torcs_base_dir:    str   = TORCS_CONFIG_DIR
    instance_dir_pattern: str = r"D:\torcs\torcs_inst{idx}"

    # Timing
    startup_wait_s:    float = 12.0   # wait after launching before connecting
    menu_settle_s:     float = 3.0    # wait after focus before sending menu keys
    restart_timeout_s: float = 30.0   # kill+restart if instance hangs this long
    poll_interval_s:   float = 0.5    # health-check polling interval

    # Eval settings per trial
    eval_laps:         int   = 3      # laps per evaluation (take best)
    eval_timeout_s:    float = 600.0  # max seconds per evaluation


# Reward Function
@dataclass
class RewardConfig:
    # Stage 1: Time Trial
    w_progress: float = 1.0
    progress_scale: float = 3602.0

    # Per-step time cost
    w_time: float = 0.1
    w_trackpos: float = -0.05
    w_angle: float = -0.05
    w_smoothness: float = -0.1
    w_speed_bonus: float = 0.2
    speed_scale: float = 300.0
    w_cornering: float = 1.0
    cornering_threshold: float = 150.0
    w_traction: float = 0.0

    # Straight detection
    straight_threshold: float = 150.0

    # Penalties
    penalty_offtrack: float = -1.0
    penalty_backwards: float = -10.0
    backwards_penalty: float = -10.0 # Alias

    # Lap completion
    lap_bonus: float = 500.0
    lap_target_time: float = 90.0

    # Damage
    w_damage: float = -0.025
    damage_cap: float = 200.0
    damage_penalty: float = -0.001  # Alias

    # Stage 2: Multi-Car
    w_opponent_proximity: float = -1.0
    opponent_danger_dist: float = 10.0
    bonus_overtake: float = 5.0
    penalty_collision: float = -5.0


# Observation
@dataclass
class ObservationConfig:
    # Normalization constants
    speed_max: float = 300.0             # km/h max for normalization
    rpm_max: float = 10000.0
    gear_max: float = 7.0                # fixed obs-norm constant (gear/7) bc_v6.pth was trained with; keep as-is
    track_sensor_max: float = 200.0      # Track edge sensor range in meters
    opponent_sensor_max: float = 200.0   # Opponent sensor range in meters
    wheel_spin_max: float = 100.0        # rad/s normalization cap
    lap_time_max: float = 120.0          # Normalize lap time (seconds)

    # Feature dimensions
    n_track_sensors: int = 19
    n_opponent_sensors: int = 36
    n_wheel_spin: int = 4
    include_prev_steer: bool = True

    @property
    def stage1_dim(self) -> int:
        # 32-dim observation for empty track (includes prev_steer)
        return (
            self.n_track_sensors +  # 19: track edge sensors
            3 +                     # speedX, speedY, angle
            1 +                     # trackPos
            1 +                     # rpm
            1 +                     # gear
            self.n_wheel_spin +     # 4: wheel spin velocities
            1 +                     # distFromStart (track progress)
            1 +                     # curLapTime
            1                       # prev_steer (last executed steering command — keeps MDP Markov)
        )  # = 32

    @property
    def stage2_dim(self) -> int:
        # 68-dim observation with opponent sensors (32 + 36)
        return self.stage1_dim + self.n_opponent_sensors


# SAC / Network
@dataclass
class SACConfig:
    # Network architecture
    pi_layers:  List[int] = field(default_factory=lambda: [256, 256, 128])
    qf_layers:  List[int] = field(default_factory=lambda: [256, 256, 128])

    # SAC hyperparameters (Stage 1 defaults — Optuna refines these)
    learning_rate:   float = 3e-4
    learning_rate_stage2: float = 1e-4
    buffer_size:     int   = 1_000_000
    batch_size:      int   = 256
    tau:             float = 0.005
    gamma:           float = 0.99
    train_freq:      int   = 1
    gradient_steps:  int   = 1
    ent_coef:        str   = "auto"
    target_entropy:  str   = "auto"
    use_sde:         bool  = True
    sde_sample_freq: int   = 8
    learning_starts: int   = 10_000

    # BC anchor
    bc_coef0: float = 100.0
    bc_decay_steps: int = 300_000
    log_std_init: float = -3.0
    freeze_steps: int = 1000

    # Device
    device: str = "auto"
    cpu_threads: int = 4


# Training Schedule
@dataclass
class TrainingConfig:
    stage1_total_timesteps: int = 5_000_000
    stage2_total_timesteps: int = 2_000_000

    # Legacy attributes for older environments
    stage1_eval_freq: int = 5000
    stage1_checkpoint_freq: int = 10_000
    stage1_target_lap_time: float = 90.0
    stage2_eval_freq: int = 5000
    stage2_checkpoint_freq: int = 10_000
    opponent_bot_module: str = "inferno"
    tensorboard_log: str = LOG_DIR
    verbose: int = 1

    # Seeding
    seed_steps:    int = 50_000   # teacher rollout steps to seed buffer

    # Checkpointing & logging
    checkpoint_freq: int = 50_000
    car_telemetry_every_n_steps:    int = 1
    neuron_telemetry_every_n_steps: int = 100
    torcs_relaunch_every_n_steps:   int = 250_000

    # Speed cap curriculum (from-scratch SAC)
    speed_cap_start_kmh:      float = 80.0
    speed_cap_increment_kmh:  float = 40.0
    speed_cap_episodes_needed: int  = 5
    speed_cap_max_kmh:        float = 350.0


# Curriculum (Stage 1 -> 2 opponent ramp)
@dataclass
class CurriculumConfig:
    stage2_n_opponents:         int = 1
    initial_opponents:          int = 1
    max_opponents:              int = 10
    opponents_increment_every:  int = 100

    # Legacy transitions
    min_completed_laps: int = 10
    max_avg_lap_time: float = 95.0
    consecutive_good_episodes: int = 5


# Residual RL (DAgger base + SAC corrections)
@dataclass
class ResidualRLConfig:
    # Path to the frozen DAgger base policy
    dagger_weights: str = os.path.join(CHECKPOINT_DIR, "dagger_policy_v2.pth")

    # Residual action boundaries (how much SAC can deviate from DAgger)
    delta_steer:    float = 0.15   # ±15% steering freedom
    delta_accel:    float = 0.50   # ±50% throttle/braking freedom

    # SAC hyperparameters
    learning_rate:   float = 1e-4
    buffer_size:     int   = 500_000
    batch_size:      int   = 256
    tau:             float = 0.005
    gamma:           float = 0.99
    train_freq:      tuple = (1, "episode") # Update weights ONLY when car is safely reset
    gradient_steps:  int   = -1             # Do as many gradient steps as physics steps were taken
    ent_coef:        float = 0.02
    use_sde:         bool  = True
    sde_sample_freq: int   = 1
    log_std_init:    float = -3.5

    # BC Anchor (L2 Regularization on residual)
    bc_coef0:        float = 100.0    # TD3+BC normalization math balances perfectly around 100.0
    bc_decay_steps:  int = 500_000 # decay to 0 over 500k steps

    # Training schedule
    total_timesteps: int = 2_000_000
    seed_steps:      int = 20_000
    freeze_steps:    int = 1_000   # critic warmup before actor unfreezes
    eval_freq:       int = 20_000
    checkpoint_freq: int = 10_000

    # gSDE actor init
    zero_init_mu:    bool  = True


# Driver Aids
@dataclass
class AidsConfig:
    # Action space
    n_actions: int = 2

    # Traction Control System
    tcs_enabled:         bool  = True
    tcs_slip_threshold:  float = 5.0    # (rear - front) wheelspin rad/s to trigger (was 0.20 — way too aggressive)
    tcs_throttle_cut:    float = 0.60   # reduce throttle to this fraction on TCS
    tcs_min_accel_factor: float = 0.1
    tcs_min_speed:       float = 30.0

    # Steering rate limiter (prevents snap oversteer)
    steer_rate_limit_enabled: bool = False
    steer_rate_limit:    float = 0.10   # max steer change per step

    # Gear shift thresholds
    up_rpm_threshold:    float = 7000.0
    down_rpm_threshold:  float = 3000.0
    rpm_upshift:         float = 17800.0  # near car1-ow1 redline (18700) — high-rev, no short-shift
    rpm_downshift:       float = 9000.0
    min_gear:            int   = 1
    max_gear:            int   = 6         # SCR clamps gear commands to -1..6; top speed is reached in 5th

    # Launch assist
    launch_assist_enabled: bool  = True
    launch_release_speed:  float = 40.0
    launch_max_time:       float = 8.0
    launch_centering_gain: float = 0.30


# ─────────────────────────────────────────────────────────────────────
#  Global Config singleton (mirrors S3 interface so all files work)
# ─────────────────────────────────────────────────────────────────────

class Config:
    torcs       = TorcsConfig()
    teacher_v3  = TeacherV3Params()
    multi       = MultiInstanceConfig()
    reward      = RewardConfig()
    observation = ObservationConfig()
    sac         = SACConfig()
    training    = TrainingConfig()
    curriculum  = CurriculumConfig()
    residual    = ResidualRLConfig()
    aids        = AidsConfig()

    # Aliases for backwards compatibility
    obs         = observation
    action      = aids

    CHECKPOINT_DIR = CHECKPOINT_DIR
    LOG_DIR        = LOG_DIR
    TELEMETRY_DIR  = TELEMETRY_DIR
    TORCS_EXE      = TORCS_EXE

    @staticmethod
    def get_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def summary() -> str:
        lines = [
            "=" * 60,
            "  S4-F Configuration Summary",
            f"  Device:       {Config.get_device()}",
            f"  Track:        {Config.torcs.track_name}  ({Config.torcs.track_length_m}m)",
            f"  TORCS:        {TORCS_EXE}",
            f"  Parallel:     {Config.multi.n_instances} instances (ports {Config.multi.base_port}..{Config.multi.base_port + Config.multi.n_instances - 1})",
            f"  Target:       <= 80s lap time",
            "=" * 60,
        ]
        return "\n".join(lines)


# TempConfig: context manager for Optuna trials
class TempConfig:
    # Temporarily override Config attributes for a trial scope

    def __init__(self, overrides: dict):
        self._overrides = overrides
        self._saved = {}

    def __enter__(self):
        for path, value in self._overrides.items():
            obj, attr = self._resolve(path)
            self._saved[path] = (obj, attr, copy.deepcopy(getattr(obj, attr)))
            setattr(obj, attr, value)
        return self

    def __exit__(self, *_):
        for path, (obj, attr, original) in self._saved.items():
            setattr(obj, attr, original)

    @staticmethod
    def _resolve(path: str):
        # Resolve 'residual.delta' -> (Config.residual, 'delta')
        parts = path.split(".")
        if len(parts) == 2:
            obj = getattr(Config, parts[0])
            return obj, parts[1]
        elif len(parts) == 3:
            obj = getattr(Config, parts[0])
            return obj, parts[1]
        raise ValueError(f"TempConfig: unsupported path format: {path!r}")

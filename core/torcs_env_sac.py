# SB3-compatible Gymnasium environment for TORCS
# Action: Box([-1,-1], [1,1]) -> [steer, accel_brake]
# Observation: Box(-1, 1, shape=(32,)) for Stage 1; (68,) for Stage 2

import os
import sys
import time
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import core.snakeoil3_gym as snakeoil3
from config import Config
from core.observation_utils import build_observation, get_observation_dim, raw_obs_to_dict_safe
from core.reward_functions import compute_reward
from core.driving_aids import AidsState, apply_aids


class TorcsSACEnv(gym.Env):
    # Gymnasium-compatible TORCS environment for SAC training

    metadata = {"render_modes": ["human"]}

    def __init__(self, stage: int = 1, render_mode: str = None, port: int = None):
        super().__init__()

        self.stage = stage
        self.render_mode = render_mode

        self._torcs_cfg  = Config.torcs
        self._action_cfg = Config.action
        self._obs_cfg    = Config.obs

        self._port = port if port is not None else self._torcs_cfg.port

        obs_dim = get_observation_dim(self.stage)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # TORCS connection
        self.client = None

        # Episode state
        self._raw_obs       = None
        self._raw_obs_prev  = None
        self._action_prev   = None
        self._current_gear  = 1
        self._time_step     = 0
        self._episode_count = 0
        self._total_reward  = 0.0
        self._initial_reset_done = False

        # Termination tracking
        self._offtrack_counter    = 0
        self._last_progress_dist  = 0.0
        self._last_progress_step  = 0
        self._last_lap_time       = None

        # Driving aids state
        self._aids_state = AidsState()
        self._prev_steer = 0.0

    def _launch_torcs(self):
        # Launch TORCS simulator process
        self._kill_torcs()
        time.sleep(0.5)

        if os.name == 'nt':
            torcs_dir = self._torcs_cfg.torcs_dir
            torcs_exe = self._torcs_cfg.torcs_exe
            cmd = f'start "" /D "{torcs_dir}" "{torcs_exe}" -nolaptime'
            if self._torcs_cfg.vision:
                cmd += " -vision"
            os.system(cmd)
            os.system(f'python "{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "autostart_win.py")}"')
        else:
            # Linux/WSL headless launch
            if not getattr(TorcsSACEnv, "_linux_warmed", False):
                os.system("DISPLAY= WAYLAND_DISPLAY= timeout 10 torcs >/dev/null 2>&1")
                TorcsSACEnv._linux_warmed = True
            from search.optuna_teacher_linux import install_practice_xml, launch_torcs
            race_xml = install_practice_xml(self._port)
            launch_torcs(race_xml, self._port, "r", ":1")

        time.sleep(0.5)

    def _kill_torcs(self):
        # Kill all running TORCS processes
        if os.name == 'nt':
            os.system("taskkill /IM wtorcs.exe /F >nul 2>&1")
            os.system("taskkill /IM torcs.exe /F >nul 2>&1")
        else:
            from search.optuna_teacher_linux import kill_torcs
            kill_torcs()

    def _connect_client(self):
        # Establish UDP connection to TORCS SCR server
        saved_argv = sys.argv
        try:
            sys.argv = [sys.argv[0]]
            self.client = snakeoil3.Client(
                p=self._port, vision=self._torcs_cfg.vision,
            )
        finally:
            sys.argv = saved_argv
        self.client.MAX_STEPS = np.inf

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._time_step    = 0
        self._total_reward = 0.0
        self._action_prev  = None
        self._current_gear = 1
        self._last_lap_time = None
        self._aids_state.reset()
        self._prev_steer   = 0.0

        self._offtrack_counter    = 0
        self._last_progress_dist  = 0.0
        self._last_progress_step  = 0

        relaunch = (
            not self._initial_reset_done
            or (self._episode_count % self._torcs_cfg.relaunch_every_n_episodes == 0)
        )

        # Launch + connect with retries
        max_tries = getattr(self._torcs_cfg, "launch_retries", 4)
        last_err = None
        for attempt in range(max_tries):
            try:
                if relaunch or attempt > 0:
                    self._launch_torcs()
                elif self._initial_reset_done:
                    try:
                        self.client.R.d["meta"] = True
                        self.client.respond_to_server()
                    except Exception:
                        pass
                self._connect_client()
                self.client.get_servers_input()
                break
            except Exception as e:
                last_err = e
                print(f"[env] TORCS launch/connect failed "
                      f"(attempt {attempt + 1}/{max_tries}): {e}", flush=True)
                self._kill_torcs()
                time.sleep(1.0)
                relaunch = True
        else:
            raise ConnectionError(f"TORCS unreachable after {max_tries} attempts: {last_err}")

        self._raw_obs      = raw_obs_to_dict_safe(self.client.S.d)
        self._raw_obs_prev = None

        self._initial_reset_done = True
        self._episode_count += 1

        obs = build_observation(self._raw_obs, self.stage, prev_steer=self._prev_steer)

        info = {
            "episode_count": self._episode_count,
            "stage": self.stage,
            "raw_obs": copy.deepcopy(self._raw_obs),
        }
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)

        # Build TORCS action via driving_aids
        cmd = apply_aids(self._raw_obs, action, self._aids_state, self._action_cfg)
        self._prev_steer   = self._aids_state.prev_steer
        self._current_gear = self._aids_state.current_gear

        torcs_action = self.client.R.d
        torcs_action["steer"] = cmd["steer"]
        torcs_action["accel"] = cmd["accel"]
        torcs_action["brake"] = cmd["brake"]
        torcs_action["gear"]  = cmd["gear"]

        self._raw_obs_prev = copy.deepcopy(self._raw_obs)

        # Simulate one step
        self.client.respond_to_server()
        self.client.get_servers_input()
        self._raw_obs = raw_obs_to_dict_safe(self.client.S.d)

        self._time_step += 1

        # Termination context flags
        track     = np.array(self._raw_obs.get("track", [0.0] * 19), dtype=np.float32)
        angle     = float(self._raw_obs.get("angle", 0.0))
        track_pos = float(self._raw_obs.get("trackPos", 0.0))

        # Off-track determined by trackPos threshold (not track sensors)
        off_track = bool(abs(track_pos) > self._torcs_cfg.offtrack_trackpos_threshold)
        backwards = bool(np.cos(angle) < 0)

        if off_track:
            self._offtrack_counter += 1
        else:
            self._offtrack_counter = 0

        # Lap completion detection
        cur_lap_time  = float(self._raw_obs.get("curLapTime", 0.0))
        last_lap_time = float(self._raw_obs.get("lastLapTime", 0.0))
        lap_completed = (last_lap_time > 0.0 and last_lap_time != self._last_lap_time)
        if lap_completed:
            self._last_lap_time = last_lap_time

        # Compute reward
        reward = compute_reward(
            self._raw_obs, self._raw_obs_prev, action, self._action_prev,
            stage=self.stage, off_track=off_track, backwards=backwards,
            lap_completed=lap_completed,
            last_lap_time=last_lap_time if lap_completed else 0.0,
        )

        # Termination: unrecoverable states
        damage = float(self._raw_obs.get("damage", 0.0))
        too_damaged = damage > self._torcs_cfg.max_damage
        terminated = backwards or too_damaged or (
            self._offtrack_counter > self._torcs_cfg.max_offtrack_steps
        )

        # Truncation: recoverable but fruitless states
        truncated = False
        dist_raced = float(self._raw_obs.get("distRaced", 0.0))
        if dist_raced - self._last_progress_dist > self._torcs_cfg.progress_stuck_min_dist:
            self._last_progress_dist = dist_raced
            self._last_progress_step = self._time_step

        steps_without_progress = self._time_step - self._last_progress_step
        if steps_without_progress > self._torcs_cfg.progress_stuck_window:
            truncated = True
            reward -= 5.0

        if self._time_step >= self._torcs_cfg.max_steps_per_episode:
            truncated = True

        # Signal TORCS to reset on episode end
        if terminated or truncated:
            self.client.R.d["meta"] = True
            self.client.respond_to_server()

        # Update state
        self._action_prev   = action.copy()
        self._total_reward += reward

        obs = build_observation(self._raw_obs, self.stage, prev_steer=self._prev_steer)

        info = {
            "episode_count":  self._episode_count,
            "step":           self._time_step,
            "stage":          self.stage,
            "speedX":         float(self._raw_obs.get("speedX", 0.0)),
            "trackPos":       track_pos,
            "angle":          angle,
            "gear":           self._current_gear,
            "rpm":            float(self._raw_obs.get("rpm", 0.0)),
            "distRaced":      dist_raced,
            "curLapTime":     cur_lap_time,
            "lastLapTime":    last_lap_time,
            "damage":         float(self._raw_obs.get("damage", 0.0)),
            "total_reward":   self._total_reward,
            "off_track":      off_track,
            "lap_completed":  lap_completed,
            "applied_steer":  self._prev_steer,
            "raw_obs":        copy.deepcopy(self._raw_obs),
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        self._kill_torcs()

    def _update_gear(self):
        # Auto gear shift (matches teacher's shift schedule)
        rpm = float(self._raw_obs.get("rpm", 0.0)) if self._raw_obs else 0.0
        g = self._current_gear if self._current_gear >= 1 else 1

        if rpm > self._action_cfg.rpm_upshift and g < self._action_cfg.max_gear:
            g += 1
        elif rpm < self._action_cfg.rpm_downshift and g > 1:
            g -= 1

        self._current_gear = g

    def get_raw_obs(self) -> dict:
        return copy.deepcopy(self._raw_obs) if self._raw_obs else {}

    def get_prev_steer(self) -> float:
        return self._prev_steer

    def set_stage(self, stage: int):
        # Switch curriculum stage (changes obs dim and reward function)
        if stage not in (1, 2):
            raise ValueError(f"Stage must be 1 or 2, got {stage}")
        self.stage = stage
        obs_dim = get_observation_dim(self.stage)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

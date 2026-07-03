# Phase 2: stage-2 residual environment (multi-car).
# Frozen 32-dim base policy (bc_v6) reads obs[:32]; the SAC residual reads the full
# 68-dim observation (32 self + 36 opponent sensors).
#   final_action = clip(base(obs[:32]) + delta * residual(obs68), -1, 1)
# The frozen base guarantees a performance floor; the residual learns opponent-aware
# corrections (avoidance / overtaking / defense).

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # phase2/  (race_config)
sys.path.insert(0, os.path.dirname(_HERE))      # project root (config, torcs_env_sac, ...)

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from config import Config
from agents.bc_pretrain import BCNetwork
from core.observation_utils import get_observation_dim
from core.torcs_env_sac import TorcsSACEnv
from phase2.race_config import install_massstart_xml


class TorcsSACEnvStage2(TorcsSACEnv):
    # Stage-2 inner env: 68-dim obs, reward_stage2, multi-car grid (headless only).
    # Deferred reset: the real race reset is delayed to the first step() so the new
    # grid isn't live during SB3's post-episode backprop — otherwise bots would
    # drive off the grid while we compute gradients. Ensures a synchronized start.

    def __init__(self, port=None, n_opponents=4, bot_module="inferno"):
        self.n_opponents = int(n_opponents)
        self.bot_module = bot_module
        self._defer_pending = False
        self._cached_obs = None
        super().__init__(stage=2, port=port)

    def set_n_opponents(self, n: int) -> None:
        # Takes effect on the next TORCS relaunch.
        self.n_opponents = int(n)

    def reset(self, seed=None, options=None):
        # First reset must be real (SB3 needs a valid initial obs); later resets
        # are deferred to the next step() — see class comment.
        if self._initial_reset_done and self._cached_obs is not None:
            self._defer_pending = True
            return self._cached_obs.copy(), {"deferred_reset": True,
                                             "raw_obs": self.get_raw_obs()}
        obs, info = super().reset(seed=seed, options=options)
        self._cached_obs = obs.copy()
        return obs, info

    def step(self, action):
        if self._defer_pending:
            # Real restart now: meta/relaunch -> scr_server blocks until we
            # reconnect -> bots held on the grid through the gradient phase.
            self._defer_pending = False
            obs, _ = super().reset()
            self._cached_obs = obs.copy()
        obs, reward, terminated, truncated, info = super().step(action)
        self._cached_obs = obs.copy()
        return obs, reward, terminated, truncated, info

    def _launch_torcs(self):
        if os.name == "nt":
            raise NotImplementedError("Stage-2 training is headless Linux/WSL only")
        self._kill_torcs()
        time.sleep(0.5)
        if not getattr(TorcsSACEnv, "_linux_warmed", False):
            os.system("DISPLAY= WAYLAND_DISPLAY= timeout 10 torcs >/dev/null 2>&1")
            TorcsSACEnv._linux_warmed = True
        from search.optuna_teacher_linux import launch_torcs
        race_xml = install_massstart_xml(self._port, self.n_opponents, self.bot_module)
        launch_torcs(race_xml, self._port, "r", ":1")
        time.sleep(0.5)


class Stage2ResidualEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, base_weights=None, n_opponents=4, bot_module="inferno",
                 port=None, device="cpu", delta=None, render_mode=None):
        super().__init__()
        cfg = Config.residual
        self.delta = np.array(delta if delta is not None
                              else [cfg.delta_steer, cfg.delta_accel], dtype=np.float32)
        self.device = device
        self.render_mode = render_mode

        self._inner = TorcsSACEnvStage2(port=port, n_opponents=n_opponents, bot_module=bot_module)
        self.observation_space = self._inner.observation_space           # 68-dim
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # Frozen base at STAGE-1 dim (32), consumes obs[:32] only.
        self._base_dim = get_observation_dim(1)
        self._base = BCNetwork(obs_dim=self._base_dim, action_dim=2, hidden_sizes=[256, 256, 128])
        default_base = os.path.join(Config.CHECKPOINT_DIR, "bc_v6.pth")
        self._load_base(base_weights or default_base)

        self._last_obs = None

    def set_n_opponents(self, n: int) -> None:
        self._inner.set_n_opponents(n)

    def _load_base(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Stage2] frozen base weights not found: {path}")
        state = torch.load(path, map_location=self.device, weights_only=True)
        w0 = state.get("net.0.weight")
        if w0 is not None and w0.shape[1] != self._base_dim:            # 31 -> 32 legacy pad
            padded = torch.zeros(w0.shape[0], self._base_dim)
            cols = min(w0.shape[1], self._base_dim)
            padded[:, :cols] = w0[:, :cols]
            state["net.0.weight"] = padded
        self._base.load_state_dict(state)
        self._base.eval()
        for p in self._base.parameters():
            p.requires_grad_(False)
        print(f"[Stage2] frozen base loaded: {path}  (reads obs[:{self._base_dim}], "
              f"delta_steer={self.delta[0]:.3f}, delta_accel={self.delta[1]:.3f})")

    def base_action(self, obs: np.ndarray) -> np.ndarray:
        obs32 = np.asarray(obs, dtype=np.float32)[: self._base_dim]
        with torch.no_grad():
            return self._base(torch.tensor(obs32).unsqueeze(0)).numpy()[0]

    def compute_final_action(self, obs: np.ndarray, residual: np.ndarray) -> np.ndarray:
        return np.clip(self.base_action(obs) + self.delta * residual, -1.0, 1.0)

    def get_raw_obs(self) -> dict:
        return self._inner.get_raw_obs()

    def reset(self, seed=None, options=None):
        obs, info = self._inner.reset(seed=seed, options=options)
        self._last_obs = obs.copy()
        return obs, info

    def step(self, residual: np.ndarray):
        residual = np.clip(residual, -1.0, 1.0)
        final = (self.compute_final_action(self._last_obs, residual)
                 if self._last_obs is not None else np.clip(residual, -1.0, 1.0))
        obs, reward, terminated, truncated, info = self._inner.step(final)
        info["residual"]     = residual.copy()
        info["final_action"] = final.copy()
        self._last_obs = obs.copy()
        return obs, reward, terminated, truncated, info

    def close(self):
        self._inner.close()

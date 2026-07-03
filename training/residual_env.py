# Residual RL environment wrapper
# SAC learns bounded corrections on top of a frozen DAgger base policy.
# final_action = clip(dagger(obs) + delta * residual, -1, 1)

import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from config import Config
from agents.bc_pretrain import BCNetwork
from core.observation_utils import get_observation_dim
from core.torcs_env_sac import TorcsSACEnv


class ResidualTorcsEnv(gym.Env):
    # TorcsSACEnv wrapper for Residual RL with frozen DAgger base

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        stage: int = 1,
        dagger_weights: str = None,
        device: str = "cpu",
        port: int = None,
        render_mode: str = None,
    ):
        super().__init__()

        cfg = Config.residual
        self.stage       = stage
        self.delta       = np.array([cfg.delta_steer, cfg.delta_accel], dtype=np.float32)
        self.device      = device
        self.render_mode = render_mode

        # Inner env handles TORCS interaction
        self._inner = TorcsSACEnv(stage=stage, port=port)

        # Spaces: same obs as inner env; SAC action is the residual
        self.observation_space = self._inner.observation_space
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # Frozen DAgger base policy
        weights_path = dagger_weights or cfg.dagger_weights
        obs_dim = get_observation_dim(stage)
        self._dagger = BCNetwork(obs_dim=obs_dim, action_dim=2, hidden_sizes=[256, 256, 128])
        self._load_dagger(weights_path)

        self._last_obs: np.ndarray = None

    def _load_dagger(self, path: str) -> None:
        # Load DAgger weights (handles 31->32 obs-dim mismatch)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[ResidualEnv] DAgger weights not found: {path}\n"
                f"Run dagger.py first to produce dagger_policy_v2.pth."
            )
        state = torch.load(path, map_location=self.device, weights_only=True)

        # Obs-dim mismatch guard (31-dim checkpoint -> 32-dim network)
        obs_dim = get_observation_dim(self.stage)
        w0 = state.get("net.0.weight")
        if w0 is not None and w0.shape[1] != obs_dim:
            print(f"[ResidualEnv] obs-dim mismatch: ckpt={w0.shape[1]}, net={obs_dim} "
                  f"— zero-padding prev_steer column.")
            padded = torch.zeros(w0.shape[0], obs_dim)
            padded[:, :w0.shape[1]] = w0
            state["net.0.weight"] = padded

        self._dagger.load_state_dict(state)
        self._dagger.eval()
        for p in self._dagger.parameters():
            p.requires_grad_(False)

        print(f"[ResidualEnv] DAgger base loaded: {path}  "
              f"(delta_steer={self.delta[0]:.3f}, delta_accel={self.delta[1]:.3f})")

    def dagger_action(self, obs: np.ndarray) -> np.ndarray:
        # Frozen DAgger forward pass on a single obs vector
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self._dagger(obs_t).numpy()[0]   # shape (2,)

    def compute_final_action(self, obs: np.ndarray, residual: np.ndarray) -> np.ndarray:
        # Combine DAgger base with SAC residual
        base = self.dagger_action(obs)
        return np.clip(base + self.delta * residual, -1.0, 1.0)

    def get_raw_obs(self) -> dict:
        return self._inner.get_raw_obs()

    # Gym API
    def reset(self, seed=None, options=None):
        obs, info = self._inner.reset(seed=seed, options=options)
        self._last_obs = obs.copy()
        return obs, info

    def step(self, residual: np.ndarray):
        # Apply residual correction to DAgger base and step inner env
        residual = np.clip(residual, -1.0, 1.0)

        if self._last_obs is not None:
            final_action = self.compute_final_action(self._last_obs, residual)
        else:
            final_action = np.clip(residual, -1.0, 1.0)

        obs, reward, terminated, truncated, info = self._inner.step(final_action)

        # Augment info for telemetry / diagnostics
        info["residual"]     = residual.copy()
        info["final_action"] = final_action.copy()
        info["dagger_base"]  = (self.dagger_action(self._last_obs)
                                if self._last_obs is not None else None)

        self._last_obs = obs.copy()
        return obs, reward, terminated, truncated, info

    def close(self):
        self._inner.close()

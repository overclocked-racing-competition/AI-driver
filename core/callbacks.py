# SB3 Custom Callbacks for SAC Training

import os
import time
import numpy as np
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from config import Config
from core.telemetry_recorder import TelemetryRecorder


class TelemetryCallback(BaseCallback):
    # Bridges TelemetryRecorder with SB3's training loop

    def __init__(
        self,
        recorder: TelemetryRecorder,
        car_freq: int = 1,
        neuron_freq: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.recorder = recorder
        self.car_freq = car_freq
        self.neuron_freq = neuron_freq
        self._episode_count = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        self._start_time = time.time()

    def _on_step(self) -> bool:
        # Get info from the environment
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}

        rewards = self.locals.get("rewards", [0.0])
        reward = float(rewards[0]) if rewards else 0.0

        actions = self.locals.get("actions", [np.zeros(3)])
        action = actions[0] if len(actions) > 0 else np.zeros(3)

        self._episode_reward += reward
        self._episode_length += 1

        # Car telemetry
        if self.num_timesteps % self.car_freq == 0:
            raw_obs = info.get("raw_obs", {})
            self.recorder.log_car(
                episode=self._episode_count,
                step=self._episode_length,
                raw_obs=raw_obs,
                action=action,
                reward=reward,
                applied_steer=info.get("applied_steer", None),
            )

        # Neuron telemetry
        if self.num_timesteps % self.neuron_freq == 0:
            # Extract NN stats from SB3 logger if available
            actor_loss = 0.0
            critic_loss = 0.0
            ent_coef_val = 0.0

            if hasattr(self.model, "logger") and self.model.logger is not None:
                # SB3 stores these in the logger's name_to_value dict
                name_to_value = getattr(self.model.logger, "name_to_value", {})
                actor_loss = name_to_value.get("train/actor_loss", 0.0)
                critic_loss = name_to_value.get("train/critic_loss", 0.0)
                ent_coef_val = name_to_value.get("train/ent_coef", 0.0)

            elapsed = time.time() - self._start_time
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0.0

            # Get current learning rate
            lr = self.model.learning_rate
            if callable(lr):
                lr = lr(1.0)  # Get the current value

            # Replay buffer size
            buf_size = 0
            if hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                buf_size = self.model.replay_buffer.size()

            self.recorder.log_neuron(
                episode=self._episode_count,
                step=self._episode_length,
                global_step=self.num_timesteps,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                entropy_coef=ent_coef_val,
                mean_reward=self._episode_reward / max(1, self._episode_length),
                episode_reward=self._episode_reward,
                episode_length=self._episode_length,
                learning_rate=lr,
                buffer_size=buf_size,
                fps=fps,
            )

        # Episode end detection
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self._episode_count += 1
            self._episode_reward = 0.0
            self._episode_length = 0
            self.recorder.new_episode(self._episode_count)

        return True

    def _on_training_end(self):
        self.recorder.close()


class LapTimeCallback(BaseCallback):
    # Tracks completed lap times and logs to TensorBoard

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._last_recorded_lap_time = None
        self._best_lap_time = float("inf")
        self._lap_times = []
        self._episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}

        last_lap_time = info.get("lastLapTime", 0.0)

        # New lap completed
        if (
            last_lap_time > 0
            and last_lap_time != self._last_recorded_lap_time
        ):
            self._last_recorded_lap_time = last_lap_time
            self._lap_times.append(last_lap_time)

            # Update best
            if last_lap_time < self._best_lap_time:
                self._best_lap_time = last_lap_time

            # Log to TensorBoard
            self.logger.record("race/lap_time", last_lap_time)
            self.logger.record("race/best_lap_time", self._best_lap_time)
            self.logger.record("race/total_laps", len(self._lap_times))

            if len(self._lap_times) >= 5:
                avg_last5 = np.mean(self._lap_times[-5:])
                self.logger.record("race/avg_lap_time_last5", avg_last5)

            if self.verbose > 0:
                print(
                    f"[Lap] Time: {last_lap_time:.2f}s | "
                    f"Best: {self._best_lap_time:.2f}s | "
                    f"Total laps: {len(self._lap_times)}"
                )

        # Per-step race data
        self.logger.record("race/speedX", info.get("speedX", 0.0))
        self.logger.record("race/trackPos", info.get("trackPos", 0.0))
        self.logger.record("race/gear", info.get("gear", 0))

        # Detect episode end
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self._episode_count += 1
            self.logger.record("race/episode", self._episode_count)
            self.logger.record("race/episode_reward", info.get("total_reward", 0.0))
            self.logger.record("race/episode_dist", info.get("distRaced", 0.0))

        return True

    @property
    def best_lap_time(self) -> float:
        return self._best_lap_time

    @property
    def lap_times(self) -> list:
        return list(self._lap_times)


class TorcsRelaunchCallback(BaseCallback):
    # Periodically relaunches TORCS (memory leak workaround)

    def __init__(self, relaunch_freq: int = None, verbose: int = 0):
        super().__init__(verbose)
        self.relaunch_freq = (
            relaunch_freq or Config.torcs.relaunch_every_n_episodes
        )
        self._episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self._episode_count += 1

            if self._episode_count % self.relaunch_freq == 0:
                if self.verbose > 0:
                    print(
                        f"[TORCS] Relaunching after episode {self._episode_count} "
                        f"(memory leak workaround)"
                    )
                # The environment's reset() method handles the actual relaunch
                # based on the episode counter. We just log it here.
                self.logger.record("torcs/relaunch_count", self._episode_count // self.relaunch_freq)

        return True


class FreezeActorCallback(BaseCallback):
    # Freezes SAC actor for the first N steps (critic warmup)

    def __init__(self, freeze_steps: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.freeze_steps = freeze_steps
        self._actor_lrs: list = []
        self._frozen = False

    def _on_training_start(self) -> None:
        if self.model.num_timesteps < self.freeze_steps:
            for pg in self.model.actor.optimizer.param_groups:
                self._actor_lrs.append(pg["lr"])
                pg["lr"] = 0.0
            self._frozen = True
            if self.verbose > 0:
                print(f"[FreezeActor] Actor frozen for first {self.freeze_steps} steps.")

    def _on_step(self) -> bool:
        if self._frozen and self.model.num_timesteps >= self.freeze_steps:
            # Clear momentum accumulated during freeze
            self.model.actor.optimizer.state.clear()
            for pg, lr in zip(self.model.actor.optimizer.param_groups, self._actor_lrs):
                pg["lr"] = lr
            self._frozen = False
            if self.verbose > 0:
                print(f"\n[FreezeActor] Unfrozen at step {self.model.num_timesteps}.")
        return True


class EnhancedCheckpointCallback(BaseCallback):
    # Enhanced model saving with metadata

    def __init__(
        self,
        save_freq: int = 10_000,
        save_path: str = None,
        name_prefix: str = "sac_racing",
        save_replay_buffer: bool = True,
        save_vecnorm: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path or Config.CHECKPOINT_DIR
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnorm = save_vecnorm

        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            base = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}",
            )

            # Save model
            model_path = base + "_steps"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"[Checkpoint] Model -> {model_path}.zip")

            # Save replay buffer
            if self.save_replay_buffer and hasattr(self.model, "replay_buffer"):
                buf_path = base + "_replay_buffer"
                self.model.save_replay_buffer(buf_path)
                if self.verbose > 0:
                    print(f"[Checkpoint] Replay buffer -> {buf_path}.pkl")

            # Save VecNormalize running statistics
            if self.save_vecnorm:
                env = self.model.get_env()
                if isinstance(env, VecNormalize):
                    vn_path = base + "_vecnorm.pkl"
                    env.save(vn_path)
                    if self.verbose > 0:
                        print(f"[Checkpoint] VecNormalize -> {vn_path}")

            # Keep a "latest" copy for easy resume
            latest_path = os.path.join(self.save_path, f"{self.name_prefix}_latest")
            self.model.save(latest_path)

        return True

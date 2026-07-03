# Standalone telemetry recorder (observer-pattern, decoupled from NN)
# Logs car state and neural network metrics to CSV files.

import os
import csv
import time
from datetime import datetime
from typing import Optional, Dict, Any

from config import Config


class TelemetryRecorder:
    # Records car telemetry and neuron telemetry to separate CSV files

    CAR_COLUMNS = [
        "timestamp",
        "episode",
        "step",
        "speedX",
        "speedY",
        "speedZ",
        "angle",
        "trackPos",
        "rpm",
        "gear",
        "distFromStart",
        "distRaced",
        "curLapTime",
        "lastLapTime",
        "damage",
        "fuel",
        "racePos",
        "z",
        # Track sensors (19)
        *[f"track_{i}" for i in range(19)],
        # Wheel spin (4)
        *[f"wheelSpin_{i}" for i in range(4)],
        # Opponent sensors (36) — only populated in Stage 2
        *[f"opponent_{i}" for i in range(36)],
        # Actions — raw policy demand and rate-limited applied value
        "action_steer",          # policy's raw steering demand (before rate limiter)
        "applied_steer",         # actual steer sent to TORCS (after rate limiter)
        "action_accel_brake",
        "action_gear_change",
        # Reward
        "reward",
        "cumulative_reward",
    ]

    NEURON_COLUMNS = [
        "timestamp",
        "episode",
        "step",
        "global_step",
        "actor_loss",
        "critic_loss",
        "entropy_coef",
        "entropy",
        "mean_q_value",
        "std_q_value",
        "mean_reward",
        "episode_reward",
        "episode_length",
        "learning_rate",
        "buffer_size",
        "fps",
    ]

    def __init__(self, session_name: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            return

        if session_name is None:
            session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")

        self.session_name = session_name

        # Car telemetry file
        self.car_filepath = os.path.normpath(os.path.join(
            Config.training.tensorboard_log, "..", "telemetry",
            "Car_telemetry", f"{session_name}.csv",
        ))

        # Neuron telemetry file
        self.neuron_filepath = os.path.normpath(os.path.join(
            Config.training.tensorboard_log, "..", "telemetry",
            "Neuron_telemetry", f"{session_name}.csv",
        ))

        # Ensure directories exist
        os.makedirs(os.path.dirname(self.car_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(self.neuron_filepath), exist_ok=True)

        # Open CSV files and write headers
        self._car_file = open(self.car_filepath, "w", newline="", encoding="utf-8")
        self._car_writer = csv.DictWriter(self._car_file, fieldnames=self.CAR_COLUMNS, extrasaction="ignore")
        self._car_writer.writeheader()

        self._neuron_file = open(self.neuron_filepath, "w", newline="", encoding="utf-8")
        self._neuron_writer = csv.DictWriter(self._neuron_file, fieldnames=self.NEURON_COLUMNS, extrasaction="ignore")
        self._neuron_writer.writeheader()

        # Cumulative reward tracker
        self._cumulative_reward = 0.0
        self._current_episode = 0

        # Batched flushing (flush every N rows to reduce disk I/O)
        self._flush_every = 500
        self._car_rows_since_flush = 0

        print(f"[Telemetry] Car telemetry  -> {self.car_filepath}")
        print(f"[Telemetry] Neuron telemetry -> {self.neuron_filepath}")

    def log_car(
        self,
        episode: int,
        step: int,
        raw_obs: dict,
        action: Any,
        reward: float,
        applied_steer: float = None,
    ) -> None:
        # Log one step of car telemetry
        if not self.enabled:
            return

        # Track episode changes for cumulative reward
        if episode != self._current_episode:
            self._cumulative_reward = 0.0
            self._current_episode = episode
        self._cumulative_reward += reward

        # Build row
        row = {
            "timestamp": time.time(),
            "episode": episode,
            "step": step,
            "speedX": raw_obs.get("speedX", 0.0),
            "speedY": raw_obs.get("speedY", 0.0),
            "speedZ": raw_obs.get("speedZ", 0.0),
            "angle": raw_obs.get("angle", 0.0),
            "trackPos": raw_obs.get("trackPos", 0.0),
            "rpm": raw_obs.get("rpm", 0.0),
            "gear": raw_obs.get("gear", 0),
            "distFromStart": raw_obs.get("distFromStart", 0.0),
            "distRaced": raw_obs.get("distRaced", 0.0),
            "curLapTime": raw_obs.get("curLapTime", 0.0),
            "lastLapTime": raw_obs.get("lastLapTime", 0.0),
            "damage": raw_obs.get("damage", 0.0),
            "fuel": raw_obs.get("fuel", 0.0),
            "racePos": raw_obs.get("racePos", 1),
            "z": raw_obs.get("z", 0.0),
            "reward": reward,
            "cumulative_reward": self._cumulative_reward,
        }

        # Track sensors
        track = raw_obs.get("track", [0.0] * 19)
        for i, val in enumerate(track[:19]):
            row[f"track_{i}"] = val

        # Wheel spin
        wsv = raw_obs.get("wheelSpinVel", [0.0] * 4)
        for i, val in enumerate(wsv[:4]):
            row[f"wheelSpin_{i}"] = val

        # Opponent sensors
        opponents = raw_obs.get("opponents", [200.0] * 36)
        for i, val in enumerate(opponents[:36]):
            row[f"opponent_{i}"] = val

        # Actions
        if action is not None:
            action_list = list(action) if hasattr(action, "__iter__") else [action]
            row["action_steer"] = action_list[0] if len(action_list) > 0 else 0.0
            row["applied_steer"] = applied_steer if applied_steer is not None else row["action_steer"]
            row["action_accel_brake"] = action_list[1] if len(action_list) > 1 else 0.0
            row["action_gear_change"] = action_list[2] if len(action_list) > 2 else 0.0

        self._car_writer.writerow(row)
        self._car_rows_since_flush += 1
        if self._car_rows_since_flush >= self._flush_every:
            self._car_file.flush()
            self._car_rows_since_flush = 0

    def log_neuron(
        self, episode: int, step: int, global_step: int,
        actor_loss: float = 0.0, critic_loss: float = 0.0,
        entropy_coef: float = 0.0, entropy: float = 0.0,
        mean_q_value: float = 0.0, std_q_value: float = 0.0,
        mean_reward: float = 0.0, episode_reward: float = 0.0,
        episode_length: int = 0, learning_rate: float = 0.0,
        buffer_size: int = 0, fps: float = 0.0,
    ) -> None:
        # Log one snapshot of neural network telemetry
        if not self.enabled:
            return

        row = {
            "timestamp": time.time(),
            "episode": episode,
            "step": step,
            "global_step": global_step,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy_coef": entropy_coef,
            "entropy": entropy,
            "mean_q_value": mean_q_value,
            "std_q_value": std_q_value,
            "mean_reward": mean_reward,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "fps": fps,
        }

        self._neuron_writer.writerow(row)
        self._neuron_file.flush()

    def new_episode(self, episode: int) -> None:
        # Signal new episode (resets cumulative reward)
        self._cumulative_reward = 0.0
        self._current_episode = episode

    def close(self) -> None:
        # Close all open file handles
        if not self.enabled:
            return
        if hasattr(self, "_car_file") and not self._car_file.closed:
            self._car_file.close()
            print(f"[Telemetry] Car telemetry saved -> {self.car_filepath}")
        if hasattr(self, "_neuron_file") and not self._neuron_file.closed:
            self._neuron_file.close()
            print(f"[Telemetry] Neuron telemetry saved -> {self.neuron_filepath}")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

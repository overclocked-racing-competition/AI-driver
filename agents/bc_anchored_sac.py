# BC-Anchored SAC (TD3+BC-style combined actor loss)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from typing import Optional


class BCAnchoredSAC(SAC):
    def __init__(
        self,
        *args,
        bc_coef0: float = 100.0,
        bc_decay_steps: int = 300_000,
        bc_normalization_alpha: float = 2.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bc_coef0 = bc_coef0
        self.bc_decay_steps = bc_decay_steps
        self.bc_normalization_alpha = bc_normalization_alpha

        self._demo_obs:  Optional[np.ndarray] = None
        self._demo_acts: Optional[np.ndarray] = None
        self._n_demo:    int = 0

    @property
    def bc_coef(self) -> float:
        progress = min(1.0, self.num_timesteps / max(1, self.bc_decay_steps))
        return self.bc_coef0 * (1.0 - progress)

    def add_demo_data(self, obs: np.ndarray, actions: np.ndarray) -> None:
        self._demo_obs  = obs.astype(np.float32)
        self._demo_acts = actions.astype(np.float32)
        self._n_demo    = len(obs)
        print(
            f"[BC-Anchor] {self._n_demo:,} demo transitions stored  "
            f"(bc_coef0={self.bc_coef0:.1f}, decay={self.bc_decay_steps:,} steps, "
            f"α={self.bc_normalization_alpha})"
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        bc_coef = self.bc_coef
        if self._n_demo == 0 or bc_coef <= 1e-6:
            super().train(gradient_steps, batch_size)
            return

        self._train_combined(gradient_steps, batch_size, bc_coef)

    def _train_combined(
        self, gradient_steps: int, batch_size: int, bc_coef: float
    ) -> None:
        self.policy.set_training_mode(True)

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, bc_losses = [], [], []

        for _ in tqdm(range(gradient_steps), desc="Backpropagation", leave=False, dynamic_ncols=True):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            if self.use_sde:
                self.actor.reset_noise(batch_size)

            actions_pi, log_prob = self.actor.action_log_prob(
                replay_data.observations
            )
            log_prob = log_prob.reshape(-1, 1)

            if (
                self.ent_coef_optimizer is not None
                and self.log_ent_coef is not None
            ):
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef
                    * (log_prob + self.target_entropy).detach()
                ).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_log_prob = next_log_prob.reshape(-1, 1)
                next_q = torch.cat(
                    self.critic_target(
                        replay_data.next_observations, next_actions
                    ),
                    dim=1,
                )
                next_q, _ = torch.min(next_q, dim=1, keepdim=True)
                rewards = replay_data.rewards.reshape(-1, 1)
                dones   = replay_data.dones.reshape(-1, 1)
                target_q = rewards + (1 - dones) * self.gamma * (
                    next_q - ent_coef * next_log_prob
                )
                target_q = target_q.reshape(-1, 1)

            current_q = self.critic(
                replay_data.observations, replay_data.actions
            )
            critic_loss = sum(
                F.mse_loss(cq.reshape(-1, 1), target_q) for cq in current_q
            )
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % getattr(self, "policy_delay", 1) == 0:
                actions_pi, log_prob = self.actor.action_log_prob(
                    replay_data.observations
                )
                log_prob = log_prob.reshape(-1, 1)

                q_pi = torch.cat(
                    self.critic(replay_data.observations, actions_pi), dim=1
                )
                min_q_pi, _ = torch.min(q_pi, dim=1, keepdim=True)

                sac_loss = (ent_coef * log_prob - min_q_pi).mean()

                q_magnitude = min_q_pi.abs().mean().detach().clamp(min=1.0)
                lam = self.bc_normalization_alpha / q_magnitude

                idx = np.random.randint(0, self._n_demo, size=batch_size)
                demo_obs_t = torch.tensor(
                    self._demo_obs[idx], dtype=torch.float32, device=self.device
                )
                demo_acts_t = torch.tensor(
                    self._demo_acts[idx], dtype=torch.float32, device=self.device
                )

                try:
                    feat = self.actor.extract_features(
                        demo_obs_t, self.actor.features_extractor
                    )
                except TypeError:
                    feat = self.actor.extract_features(demo_obs_t)
                latent = self.actor.latent_pi(feat)
                mean_actions = self.actor.mu(latent)
                
                if isinstance(self.actor.log_std, torch.nn.Linear):
                    log_std_tensor = self.actor.log_std(latent)
                else:
                    log_std_tensor = self.actor.log_std
                
                bc_loss = torch.mean(torch.tanh(mean_actions)**2)

                actor_loss = lam * sac_loss + bc_coef * bc_loss

                actor_losses.append(sac_loss.item())
                bc_losses.append(bc_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=1.0
                )
                self.actor.optimizer.step()

            polyak_update(
                self.critic.parameters(),
                self.critic_target.parameters(),
                self.tau,
            )
            if hasattr(self, "critic_batch_norm_stats"):
                polyak_update(
                    self.critic_batch_norm_stats,
                    self.critic_target_batch_norm_stats,
                    1.0,
                )

        self.policy.set_training_mode(False)

        logger = getattr(self, "_logger", None)
        if logger is not None:
            self.logger.record(
                "train/n_updates", self._n_updates, exclude="tensorboard"
            )
            if ent_coefs:
                self.logger.record("train/ent_coef", np.mean(ent_coefs))
            if actor_losses:
                self.logger.record("train/actor_loss", np.mean(actor_losses))
            if critic_losses:
                self.logger.record("train/critic_loss", np.mean(critic_losses))
            if bc_losses:
                self.logger.record("train/bc_loss_raw", np.mean(bc_losses))
                self.logger.record("train/bc_coef", bc_coef)
            if ent_coef_losses:
                self.logger.record(
                    "train/ent_coef_loss", np.mean(ent_coef_losses)
                )

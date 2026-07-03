# LayerNorm SAC Policy for training stability
# Adds LayerNorm after each hidden layer in both actor and critic
# to prevent Q-value overestimation and critic loss explosions.

import torch.nn as nn
from typing import List, Optional, Type

from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim


def _build_layernorm_mlp(
    input_dim: int,
    hidden_sizes: List[int],
    activation_fn: Type[nn.Module],
) -> nn.Sequential:
    # Build hidden MLP with LayerNorm: Linear -> LayerNorm -> Activation
    layers: List[nn.Module] = []
    last = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(nn.LayerNorm(h))
        layers.append(activation_fn())
        last = h
    return nn.Sequential(*layers)


class LayerNormActor(Actor):
    # SAC Actor with LayerNorm in the latent MLP

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # super() already built latent_pi (standard) plus mu/log_std heads.
        # Replace latent_pi with a LayerNorm version. Output dim is unchanged
        # (net_arch[-1]), so the mu/log_std heads remain compatible.
        self.latent_pi = _build_layernorm_mlp(
            self.features_dim, list(self.net_arch), self.activation_fn
        )


class LayerNormContinuousCritic(ContinuousCritic):
    # SAC double-Q critic with LayerNorm in each Q-network

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ContinuousCritic does not persist these, recover them.
        features_dim  = self.features_extractor.features_dim
        net_arch      = kwargs.get("net_arch") or (args[2] if len(args) > 2 else [])
        activation_fn = kwargs.get("activation_fn", nn.ReLU)
        action_dim    = get_action_dim(self.action_space)
        input_dim     = features_dim + action_dim
        last_dim      = net_arch[-1] if len(net_arch) > 0 else input_dim

        self.q_networks = []
        for idx in range(self.n_critics):
            hidden = _build_layernorm_mlp(input_dim, list(net_arch), activation_fn)
            q_net = nn.Sequential(*(list(hidden) + [nn.Linear(last_dim, 1)]))
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class LayerNormSACPolicy(SACPolicy):
    # SACPolicy using LayerNorm actor and critic

    def make_actor(
        self, features_extractor: Optional[nn.Module] = None
    ) -> LayerNormActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return LayerNormActor(**actor_kwargs).to(self.device)

    def make_critic(
        self, features_extractor: Optional[nn.Module] = None
    ) -> LayerNormContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return LayerNormContinuousCritic(**critic_kwargs).to(self.device)

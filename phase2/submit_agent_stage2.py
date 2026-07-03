# Phase 2 submission: opponent-aware agent over the SCR UDP protocol.
#   final_action = clip(base(obs[:32]) + delta * residual(obs68), -1, 1)
# Loads the frozen Phase-1 base (BCNetwork, 32-dim) and the trained residual
# (SB3 SAC, 68-dim), applies the shared driving_aids, sends commands to TORCS.
# Requires a running multi-car TORCS race with the SCR server (default port 3001).
#
# Usage: python phase2/submit_agent_stage2.py \
#            --base checkpoints/bc_v6.pth --residual checkpoints/massstart_sac_final.zip

import os
import sys
import argparse
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np
import torch
from stable_baselines3 import SAC

import core.snakeoil3_gym as snakeoil3
from config import Config
from core.observation_utils import build_observation, raw_obs_to_dict_safe, get_observation_dim
from agents.bc_pretrain import BCNetwork
from core.driving_aids import AidsState, apply_aids
from core.custom_policy import LayerNormSACPolicy


def load_base(path: str, device: str = "cpu") -> BCNetwork:
    net = BCNetwork(obs_dim=get_observation_dim(1), action_dim=2, hidden_sizes=[256, 256, 128])
    state = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser(description="IBM AI Racing League — Phase 2 (mass start) agent")
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--base", default=os.path.join(Config.CHECKPOINT_DIR, "bc_v6.pth"))
    ap.add_argument("--residual", default=os.path.join(Config.CHECKPOINT_DIR, "massstart_sac_final.zip"))
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    base = load_base(args.base)
    residual = SAC.load(args.residual, device="cpu",
                        custom_objects={"policy_class": LayerNormSACPolicy})
    delta = np.array([Config.residual.delta_steer, Config.residual.delta_accel], dtype=np.float32)
    base_dim = get_observation_dim(1)

    print(f"\n{'='*60}\n  Phase 2 agent — base + residual")
    print(f"  base:     {args.base}")
    print(f"  residual: {args.residual}\n{'='*60}")

    saved = sys.argv
    sys.argv = [sys.argv[0]]
    client = snakeoil3.Client(p=args.port, vision=Config.torcs.vision)
    sys.argv = saved

    for ep in range(args.episodes):
        aids_state = AidsState()
        prev_steer = 0.0
        max_dist, step = 0.0, 0
        client.get_servers_input()

        while step < Config.torcs.max_steps_per_episode:
            raw_obs = raw_obs_to_dict_safe(client.S.d)
            obs68 = build_observation(raw_obs, stage=2, prev_steer=prev_steer)

            with torch.no_grad():
                base_a = base(torch.tensor(obs68[:base_dim]).unsqueeze(0)).numpy()[0]
            res, _ = residual.predict(obs68, deterministic=True)
            final = np.clip(base_a + delta * res, -1.0, 1.0)

            cmd = apply_aids(raw_obs, final, aids_state)
            prev_steer = aids_state.prev_steer
            client.R.d["steer"] = cmd["steer"]
            client.R.d["accel"] = cmd["accel"]
            client.R.d["brake"] = cmd["brake"]
            client.R.d["gear"]  = cmd["gear"]
            client.respond_to_server()
            client.get_servers_input()

            step += 1
            dist = float(raw_obs.get("distRaced", 0.0))
            max_dist = max(max_dist, dist)

            angle     = float(raw_obs.get("angle", 0.0))
            track_pos = float(raw_obs.get("trackPos", 0.0))
            damage    = float(raw_obs.get("damage", 0.0))
            if (np.cos(angle) < 0 or damage > Config.torcs.max_damage
                    or abs(track_pos) > Config.torcs.offtrack_trackpos_threshold):
                client.R.d["meta"] = True
                client.respond_to_server()
                break

        print(f"  Ep {ep + 1}: dist={max_dist:.0f}m  steps={step}")


if __name__ == "__main__":
    main()

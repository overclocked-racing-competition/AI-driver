# Competition submission agent over the SCR UDP protocol.
# Two selectable policies (same driving_aids, identical to training):
#   BC (default):  a distilled BCNetwork (.pth) — deterministic forward pass.
#   Residual:      --residual <sac.zip> --base <pth> → clip(base + delta*residual).
# Pipeline: TORCS <-UDP-> snakeoil3 -> policy -> driving_aids -> commands.
# Requires a running TORCS with the SCR server (default port 3001).
#
# Usage:
#   python -m core.submit_agent [--weights bc.pth] [--port N] [--episodes N]
#   python -m core.submit_agent --residual checkpoints/residual_sac_latest.zip \
#                               --base checkpoints/dagger_policy_v2.pth

import sys
import os
import argparse
import time
import numpy as np
import torch

# Resolve sibling imports regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.snakeoil3_gym as snakeoil3
from config import Config
from core.observation_utils import build_observation, raw_obs_to_dict_safe, get_observation_dim
from agents.bc_pretrain import BCNetwork
from core.driving_aids import AidsState, apply_aids

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")


def _load_bc_net(weights_path: str, device: str = "cpu") -> BCNetwork:
    # Load a BCNetwork state dict (zero-pads a legacy 31-dim first layer to 32).
    obs_dim = get_observation_dim(stage=1)
    net = BCNetwork(obs_dim=obs_dim, action_dim=2, hidden_sizes=[256, 256, 128])
    state = torch.load(weights_path, map_location=device, weights_only=True)
    w0 = state.get("net.0.weight")
    if w0 is not None and w0.shape[1] != obs_dim:
        padded = torch.zeros(w0.shape[0], obs_dim)
        padded[:, :w0.shape[1]] = w0
        state["net.0.weight"] = padded
    net.load_state_dict(state)
    net.eval()
    return net


def make_bc_policy(weights_path: str, device: str = "cpu"):
    # Returns a callable: obs_vec(32,) -> action(2,) in [-1, 1].
    net = _load_bc_net(weights_path, device)
    print(f"[submit] Policy: BC network  ({weights_path})")

    def policy(obs_vec: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
            return net(obs_t).numpy()[0]
    return policy


def make_residual_policy(residual_zip: str, base_path: str, device: str = "cpu"):
    # Residual: final = clip(base(obs) + delta * residual(obs), -1, 1), matching
    # ResidualTorcsEnv. `base` must be the frozen net the residual was trained on.
    from stable_baselines3 import SAC
    from core.custom_policy import LayerNormSACPolicy  # custom_objects fixes the pickled path

    base = _load_bc_net(base_path, device)
    for p in base.parameters():
        p.requires_grad_(False)
    model = SAC.load(residual_zip, device=device,
                     custom_objects={"policy_class": LayerNormSACPolicy})
    delta = np.array([Config.residual.delta_steer, Config.residual.delta_accel],
                     dtype=np.float32)
    print(f"[submit] Policy: residual SAC  ({residual_zip})")
    print(f"[submit]   base={base_path}  delta={delta.tolist()}")

    def policy(obs_vec: np.ndarray) -> np.ndarray:
        residual, _ = model.predict(obs_vec, deterministic=True)
        with torch.no_grad():
            base_a = base(torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)).numpy()[0]
        return np.clip(base_a + delta * residual, -1.0, 1.0)
    return policy


def run_episode(policy, client: snakeoil3.Client, episode_num: int,
                verbose: bool = False) -> dict:
    # Drive one episode. Caller must call client.get_servers_input() first.
    aids_state = AidsState()
    prev_steer = 0.0
    max_dist   = 0.0
    lap_times: list = []
    last_lap   = None
    step       = 0
    max_steps  = Config.torcs.max_steps_per_episode
    t0         = time.time()

    print(f"\n--- Episode {episode_num} ---")

    while step < max_steps:
        raw_obs = raw_obs_to_dict_safe(client.S.d)
        obs_vec = build_observation(raw_obs, stage=1, prev_steer=prev_steer)

        nn_action = policy(obs_vec)   # [steer, accel_brake] in [-1, 1]

        cmd = apply_aids(raw_obs, nn_action, aids_state)
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

        llt = float(raw_obs.get("lastLapTime", 0.0))
        if llt > 0 and llt != last_lap:
            last_lap = llt
            lap_times.append(llt)
            print(f"  [Lap {len(lap_times)}]  {llt:.3f}s  at step {step:,}")

        if verbose and step % 100 == 0:
            print(f"  step {step:5d} | dist {dist:7.1f}m | "
                  f"spd {float(raw_obs.get('speedX',0)):6.1f} km/h | "
                  f"gear {int(raw_obs.get('gear',0))} | steer {cmd['steer']:+.3f}")

        # Termination: same predicates as the training env
        angle     = float(raw_obs.get("angle", 0.0))
        track_pos = float(raw_obs.get("trackPos", 0.0))
        damage    = float(raw_obs.get("damage", 0.0))
        backwards = np.cos(angle) < 0
        off_edge  = abs(track_pos) > Config.torcs.offtrack_trackpos_threshold
        too_dmgd  = damage > Config.torcs.max_damage

        if backwards or too_dmgd or off_edge:
            print(f"  [End] {'backwards' if backwards else 'damage' if too_dmgd else 'off-track'} "
                  f"at {dist:.0f}m step {step}")
            client.R.d["meta"] = True
            client.respond_to_server()
            break

    elapsed = time.time() - t0
    best = f"{min(lap_times):.3f}s" if lap_times else "—"
    print(f"  Dist: {max_dist:.1f}m | Laps: {len(lap_times)} | "
          f"Best: {best} | Steps: {step} | {elapsed:.0f}s")
    return {"max_dist": max_dist, "lap_times": lap_times, "steps": step}


def main():
    ap = argparse.ArgumentParser(
        description="IBM AI Racing League — neural-network submission client"
    )
    ap.add_argument("--port",     type=int, default=3001,
                    help="TORCS SCR UDP port (default: 3001)")
    ap.add_argument("--weights",  type=str, default=os.path.join(CKPT_DIR, "bc_v6.pth"),
                    help="BC policy .pth (used when --residual is not given)")
    ap.add_argument("--residual", type=str, default=None,
                    help="Run a residual SAC checkpoint (.zip) instead of BC")
    ap.add_argument("--base",     type=str, default=os.path.join(CKPT_DIR, "bc_v6.pth"),
                    help="Frozen base .pth the residual was trained on (default: bc_v6.pth)")
    ap.add_argument("--episodes", type=int, default=3, help="Episodes to run (default: 3)")
    ap.add_argument("--device",   type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--verbose",  action="store_true", help="Print per-100-step telemetry")
    args = ap.parse_args()

    mode = "residual" if args.residual else "BC"
    print(f"\n{'='*60}")
    print(f"  IBM AI Racing League — Neural Network Agent")
    print(f"  Track:    {Config.torcs.track_name} (Laguna Seca Corkscrew)")
    print(f"  Mode:     {mode}")
    print(f"  Port:     {args.port}   Episodes: {args.episodes}")
    print(f"{'='*60}")

    if args.residual:
        policy = make_residual_policy(args.residual, args.base, args.device)
    else:
        policy = make_bc_policy(args.weights, args.device)

    all_laps, all_dists = [], []
    for ep in range(1, args.episodes + 1):
        # Connect to TORCS (hide our custom args from snakeoil's getopt)
        saved_argv = sys.argv
        try:
            sys.argv = [sys.argv[0]]
            client = snakeoil3.Client(p=args.port, vision=Config.torcs.vision)
        finally:
            sys.argv = saved_argv

        client.MAX_STEPS = np.inf
        client.get_servers_input()

        stats = run_episode(policy, client, ep, verbose=args.verbose)
        all_laps.extend(stats["lap_times"])
        all_dists.append(stats["max_dist"])

    print(f"\n{'='*60}")
    print(f"  Summary — {args.episodes} episode(s)  [{mode}]")
    print(f"  Total laps:   {len(all_laps)}")
    print(f"  Avg distance: {np.mean(all_dists):.1f}m")
    if all_laps:
        print(f"  Best lap:     {min(all_laps):.3f}s")
        print(f"  Avg lap:      {np.mean(all_laps):.3f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

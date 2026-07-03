# DAgger (Dataset Aggregation) for SAC Racing AI
# Iteratively improves BC policy by querying teacher at learner-visited states.

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from config import Config, PROJECT_ROOT
from core.torcs_env_sac import TorcsSACEnv
from core.observation_utils import get_observation_dim
from agents.bc_pretrain import BCNetwork, train_bc, build_teacher


def rollout_learner_query_teacher(
    policy: BCNetwork,
    teacher,
    n_steps: int,
    stage: int = 1,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> tuple:
    # Roll out learner policy, query teacher at each state for labels
    env = TorcsSACEnv(stage=stage)
    obs_dim = get_observation_dim(stage)

    observations    = np.zeros((n_steps, obs_dim), dtype=np.float32)
    expert_actions  = np.zeros((n_steps, 2),       dtype=np.float32)

    laps     = 0
    max_dist = 0.0
    t0       = time.time()

    policy.eval()
    obs_vec, info = env.reset()
    raw_obs = info.get("raw_obs", {})
    teacher.reset()

    for step in range(n_steps):
        # Record state and teacher's label for this state
        observations[step]   = obs_vec
        expert_actions[step] = teacher.act(raw_obs)

        # Learner drives (key DAgger difference from BC)
        with torch.no_grad():
            obs_t  = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(device)
            action = policy(obs_t).cpu().numpy()[0]

        obs_vec, _, terminated, truncated, info = env.step(action)
        raw_obs = info.get("raw_obs", {})

        dist = float(raw_obs.get("distRaced", 0.0))
        max_dist = max(max_dist, dist)
        if info.get("lap_completed", False):
            laps += 1
            llt = float(info.get("lastLapTime", 0.0))
            if verbose:
                print(f"  [DAgger rollout] LAP {laps}: {llt:.2f}s")

        if terminated or truncated:
            obs_vec, info = env.reset()
            raw_obs = info.get("raw_obs", {})
            teacher.reset()

        if verbose and (step + 1) % 10000 == 0:
            elapsed = time.time() - t0
            fps = (step + 1) / max(1.0, elapsed)
            print(f"  [DAgger rollout] {step+1:,}/{n_steps:,} | "
                  f"fps: {fps:.0f} | max_dist: {max_dist:.0f}m | laps: {laps}")

    env.close()

    return observations, expert_actions, {
        "laps": laps,
        "max_dist": max_dist,
        "fps": n_steps / max(1.0, time.time() - t0),
    }


def evaluate_policy_laps(
    policy: BCNetwork,
    n_episodes: int = 3,
    stage: int = 1,
    device: torch.device = torch.device("cpu"),
) -> dict:
    # Run policy deterministically and collect lap times
    env = TorcsSACEnv(stage=stage)
    policy.eval()
    lap_times = []
    max_dist  = 0.0

    for ep in range(n_episodes):
        obs_vec, info = env.reset()
        ep_max = 0.0
        for _ in range(Config.torcs.max_steps_per_episode):
            with torch.no_grad():
                obs_t  = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(device)
                action = policy(obs_t).cpu().numpy()[0]
            obs_vec, _, terminated, truncated, info = env.step(action)
            dist = float(info.get("distRaced", 0.0))
            ep_max = max(ep_max, dist)
            if info.get("lap_completed", False):
                llt = float(info.get("lastLapTime", 0.0))
                if llt > 0:
                    lap_times.append(llt)
            if terminated or truncated:
                break
        max_dist = max(max_dist, ep_max)

    env.close()
    return {
        "best_lap": min(lap_times) if lap_times else float("inf"),
        "avg_lap":  float(np.mean(lap_times)) if lap_times else float("inf"),
        "n_laps":   len(lap_times),
        "max_dist": max_dist,
    }


def run_dagger(
    bc_weights_path: str,
    teacher,
    output_path: str,
    n_iterations: int = 3,
    steps_per_iter: int = 50_000,
    seed_steps: int = 100_000,
    n_epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 5e-4,
    stage: int = 1,
    device_str: str = "auto",
    verbose: bool = True,
):
    # Run DAgger: iteratively improve BC policy using teacher-labeled data
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    obs_dim = get_observation_dim(stage)

    # Load initial BC policy
    policy = BCNetwork(obs_dim).to(device)
    if os.path.exists(bc_weights_path):
        state = torch.load(bc_weights_path, map_location=device, weights_only=True)
        policy.load_state_dict(state)
        print(f"[DAgger] Loaded BC weights from: {bc_weights_path}")
    else:
        print(f"[DAgger] WARNING: {bc_weights_path} not found — starting from random weights.")

    # Seed aggregate with teacher demos for full-track coverage
    all_obs     = np.zeros((0, obs_dim), dtype=np.float32)
    all_actions = np.zeros((0, 2),       dtype=np.float32)
    if seed_steps > 0:
        from agents.bc_pretrain import collect_teacher_data
        print(f"[DAgger] Seeding aggregate with {seed_steps:,} teacher demo steps...")
        seed_obs, seed_acts, seed_stats = collect_teacher_data(
            seed_steps, teacher, stage=stage, verbose=verbose)
        print(f"[DAgger] Seed: {seed_stats['laps']} laps | "
              f"max_dist={seed_stats['max_dist']:.0f}m")
        all_obs     = seed_obs
        all_actions = seed_acts

    best_lap_time = float("inf")
    best_state    = policy.state_dict()

    print(f"\n{'='*60}")
    print(f"  DAgger Iterative Improvement")
    print(f"  Iterations: {n_iterations}  |  Steps/iter: {steps_per_iter:,}")
    print(f"  Device: {device}  |  obs_dim: {obs_dim}")
    print(f"{'='*60}\n")

    for iteration in range(n_iterations):
        print(f"\n--- DAgger Iteration {iteration + 1}/{n_iterations} ---")
        print(f"  Rolling out learner policy, querying teacher at each state...")

        new_obs, new_acts, rollout_stats = rollout_learner_query_teacher(
            policy=policy,
            teacher=teacher,
            n_steps=steps_per_iter,
            stage=stage,
            device=device,
            verbose=verbose,
        )

        print(f"  Rollout: {rollout_stats['laps']} laps | "
              f"max_dist={rollout_stats['max_dist']:.0f}m | "
              f"{rollout_stats['fps']:.0f} fps")

        # Aggregate
        all_obs     = np.concatenate([all_obs,     new_obs],  axis=0)
        all_actions = np.concatenate([all_actions, new_acts], axis=0)
        print(f"  Dataset size: {len(all_obs):,} total transitions")

        # Retrain policy on aggregated dataset
        print(f"  Retraining BC on aggregated dataset...")
        policy = train_bc(
            observations=all_obs,
            actions=all_actions,
            obs_dim=obs_dim,
            action_dim=2,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device_str,
            verbose=verbose,
        )

        # Evaluate
        print(f"  Evaluating policy (3 episodes)...")
        eval_stats = evaluate_policy_laps(policy, n_episodes=3, stage=stage, device=device)
        print(f"  Eval: best={eval_stats['best_lap']:.2f}s | "
              f"avg={eval_stats['avg_lap']:.2f}s | laps={eval_stats['n_laps']}")

        if eval_stats["best_lap"] < best_lap_time:
            best_lap_time = eval_stats["best_lap"]
            best_state    = {k: v.clone() for k, v in policy.state_dict().items()}
            torch.save(best_state, output_path)
            print(f"  New best lap {best_lap_time:.2f}s — saved to {output_path}")

    # Restore and save best
    policy.load_state_dict(best_state)
    torch.save(best_state, output_path)

    print(f"\n{'='*60}")
    print(f"  DAgger complete.")
    print(f"  Best lap: {best_lap_time:.2f}s")
    print(f"  Policy saved -> {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="DAgger for SAC Racing AI")
    parser.add_argument("--bc-weights", type=str,
                        default="checkpoints/bc_pretrained.pth",
                        help="Path to BC-pretrained weights (output of bc_pretrain.py)")
    parser.add_argument("--teacher-params", type=str, default=None,
                        help="Path to JSON from tune_teacher.py --mode export. "
                             "If omitted, uses default TeacherParams.")
    parser.add_argument("--controller", type=str, default="v6",
                        choices=["v1", "v2", "v3", "v6"],
                        help="Which teacher controller labels the states (default: v6). "
                             "Must match the controller the params were tuned with.")
    parser.add_argument("--output", type=str,
                        default="checkpoints/dagger_policy.pth",
                        help="Output path for final DAgger policy weights")
    parser.add_argument("--iterations", type=int, default=5,
                        help="DAgger iterations (default: 5 for v3 teacher)")
    parser.add_argument("--steps-per-iter", type=int, default=100_000,
                        help="Rollout steps per iteration (default: 100000)")
    parser.add_argument("--seed-steps", type=int, default=100_000,
                        help="Teacher-demo steps used to seed the DAgger aggregate "
                             "for full-track coverage (0 disables).")
    parser.add_argument("--epochs", type=int, default=30,
                        help="BC training epochs per iteration (default: 30)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    output_path = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    teacher = build_teacher(args.controller, args.teacher_params)

    run_dagger(
        bc_weights_path=os.path.join(PROJECT_ROOT, args.bc_weights),
        teacher=teacher,
        output_path=output_path,
        n_iterations=args.iterations,
        steps_per_iter=args.steps_per_iter,
        seed_steps=args.seed_steps,
        n_epochs=args.epochs,
        device_str=args.device,
        verbose=True,
    )


if __name__ == "__main__":
    main()

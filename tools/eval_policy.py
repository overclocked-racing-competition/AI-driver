# Deterministic vs stochastic policy evaluation (diagnostic)
# Loads BC/DAgger weights into SAC actor and runs in TORCS

import argparse
import numpy as np
import torch

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import Config
from core.custom_policy import LayerNormSACPolicy
from agents.bc_anchored_sac import BCAnchoredSAC
from core.torcs_env_sac import TorcsSACEnv
from training.train_sac import load_bc_weights


def build_model(bc_path, device):
    raw_env = DummyVecEnv([lambda: TorcsSACEnv(stage=1)])
    env = VecNormalize(raw_env, norm_obs=False, norm_reward=False,
                       clip_reward=10.0, gamma=Config.sac.gamma)
    model = BCAnchoredSAC(
        policy=LayerNormSACPolicy, env=env,
        learning_rate=Config.sac.learning_rate, buffer_size=1000,
        policy_kwargs=dict(net_arch=dict(pi=Config.sac.pi_layers, qf=Config.sac.qf_layers)),
        verbose=0, device=device,
    )
    load_bc_weights(model, bc_path, verbose=1)
    return model, env


def run_episode(model, env, deterministic, show_launch=False, max_steps=9000):
    raw_env = env.venv.envs[0]
    obs = env.reset()
    launch_step = None
    max_spd = 0.0
    max_dist = 0.0
    laps = 0
    last_lap = None
    if show_launch:
        print(f"    {'st':>3} {'spd':>6} {'rpm':>6} {'gr':>2} {'accel_brk':>9} {'steer':>7} {'dist':>6}")
    lap_times = []
    last_lap_recorded = None
    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rew, done, infos = env.step(action)
        info = infos[0]
        spd = float(info.get("speedX", 0.0))
        dist = float(info.get("distRaced", 0.0))
        max_spd = max(max_spd, spd)
        max_dist = max(max_dist, dist)
        if launch_step is None and spd > 30.0:
            launch_step = t
        if info.get("lap_completed", False):
            laps += 1
            llt = float(info.get("lastLapTime", 0.0))
            if llt > 0 and llt != last_lap_recorded:
                last_lap_recorded = llt
                lap_times.append(llt)
                print(f"    *** LAP {laps}: {llt:.3f}s at dist {dist:.0f}m step {t} ***")
        if show_launch and t < 30:
            rawo = info.get("raw_obs", {})
            print(f"    {t:>3} {spd:6.1f} {float(rawo.get('rpm',0)):6.0f} "
                  f"{int(rawo.get('gear',0)):>2} {float(action[0][1]):9.3f} "
                  f"{float(action[0][0]):7.3f} {dist:6.1f}")
        if done[0]:
            break
    return dict(launch_step=launch_step, max_spd=max_spd, max_dist=max_dist,
                laps=laps, lap_times=lap_times, steps=t + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-pretrain", default="checkpoints/dagger_policy.pth")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"\n{'='*64}")
    print(f"  Policy Eval: deterministic vs stochastic")
    print(f"  Weights: {args.bc_pretrain}")
    print(f"{'='*64}")

    model, env = build_model(args.bc_pretrain, args.device)

    for mode, det in [("DETERMINISTIC", True)]:
        print(f"\n--- {mode} ---")
        for ep in range(args.episodes):
            r = run_episode(model, env, deterministic=det, show_launch=(ep == 0))
            launch = f"step {r['launch_step']}" if r['launch_step'] is not None \
                     else "NEVER LAUNCHED (<30km/h)"
            lap_str = (f"  LAPS: {r['laps']}  best={min(r['lap_times']):.2f}s"
                       if r['laps'] > 0 else "  no lap")
            print(f"  Ep {ep+1}: max_dist={r['max_dist']:.0f}m  "
                  f"max_spd={r['max_spd']:.0f}km/h  "
                  f"launched={launch}  ep_len={r['steps']}"
                  f"{lap_str}")

    env.close()
    print()


if __name__ == "__main__":
    main()

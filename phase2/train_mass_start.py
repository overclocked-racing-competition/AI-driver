# Phase 2: mass-start residual SAC trainer.
# Learns an opponent-aware residual on top of the frozen bc_v6 base, with an
# opponent-count curriculum. Reuses create_model / seed_replay_buffer from
# train_residual_sac (env-agnostic), so the gentle gSDE init and residual=0
# seeding are identical to the proven Phase-1 residual stage.
#
# Usage (headless WSL, base = fast Phase-1 network):
#   HOME=/home/user/torcs_w0 python3 phase2/train_mass_start.py \
#       --base-weights checkpoints/bc_v6.pth --timesteps 1000000

import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # phase2/
sys.path.insert(0, os.path.dirname(_HERE))      # project root

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback

from config import Config
from phase2.residual_env_stage2 import Stage2ResidualEnv
from training.train_residual_sac import create_model, seed_replay_buffer

CKPT_PREFIX = "massstart_sac"


class OpponentRampCallback(BaseCallback):
    # Curriculum: raise the opponent count by 1 every `ramp_every` steps up to `maximum`.
    # The new count takes effect on the next TORCS relaunch.
    def __init__(self, start: int, maximum: int, ramp_every: int, verbose: int = 1):
        super().__init__(verbose)
        self.n = start
        self.maximum = maximum
        self.ramp_every = ramp_every
        self._next = ramp_every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next and self.n < self.maximum:
            self.n += 1
            self._next += self.ramp_every
            try:
                self.training_env.env_method("set_n_opponents", self.n)
                if self.verbose:
                    print(f"[ramp] opponents -> {self.n} (applies on next relaunch)", flush=True)
            except Exception as e:
                if self.verbose:
                    print(f"[ramp] set_n_opponents failed: {e}", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser(description="Phase 2 — mass-start residual SAC")
    ap.add_argument("--base-weights", default=os.path.join(Config.CHECKPOINT_DIR, "bc_v6.pth"),
                    help="Frozen Phase-1 base network (32-dim), read on obs[:32]")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--seed-steps", type=int, default=20_000,
                    help="residual=0 replay-buffer seeding steps")
    ap.add_argument("--start-opponents", type=int, default=2)
    ap.add_argument("--max-opponents", type=int, default=8)
    ap.add_argument("--ramp-every", type=int, default=150_000)
    ap.add_argument("--bot", default="inferno", help="TORCS bot module for opponents")
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--delta-steer", type=float, default=None,
                    help="Residual steering authority (default Config.residual; raise for swerving)")
    ap.add_argument("--delta-accel", type=float, default=None,
                    help="Residual throttle/brake authority (default Config.residual)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    dev = "cpu" if args.device in ("auto", "cpu") else args.device
    delta = None
    if args.delta_steer is not None or args.delta_accel is not None:
        c = Config.residual
        delta = [args.delta_steer if args.delta_steer is not None else c.delta_steer,
                 args.delta_accel if args.delta_accel is not None else c.delta_accel]

    def make_env():
        return Stage2ResidualEnv(
            base_weights=args.base_weights,
            n_opponents=args.start_opponents,
            bot_module=args.bot,
            port=args.port,
            device=dev,
            delta=delta,
        )

    raw = DummyVecEnv([make_env])
    env = VecNormalize(raw, norm_obs=False, norm_reward=False,
                       clip_reward=10.0, gamma=Config.residual.gamma)

    print(f"\n{'='*60}\n  Phase 2 — Mass-Start Residual SAC")
    print(f"  Base:        {args.base_weights}  (frozen, obs[:32])")
    print(f"  Opponents:   {args.start_opponents} -> {args.max_opponents} "
          f"(+1 every {args.ramp_every:,} steps, bot={args.bot})")
    print(f"  Timesteps:   {args.timesteps:,}\n{'='*60}\n")

    model = create_model(env, args)
    seed_replay_buffer(model, env, args.seed_steps, verbose=args.verbose)

    callbacks = CallbackList([
        OpponentRampCallback(args.start_opponents, args.max_opponents, args.ramp_every),
        CheckpointCallback(save_freq=10_000, save_path=Config.CHECKPOINT_DIR,
                           name_prefix=CKPT_PREFIX, save_replay_buffer=False,
                           save_vecnormalize=True),
    ])

    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                tb_log_name=CKPT_PREFIX, reset_num_timesteps=True)

    final_path = os.path.join(Config.CHECKPOINT_DIR, f"{CKPT_PREFIX}_final")
    model.save(final_path)
    env.save(os.path.join(Config.CHECKPOINT_DIR, f"{CKPT_PREFIX}_final_vecnorm.pkl"))
    print(f"\n[massstart] done -> {final_path}.zip")


if __name__ == "__main__":
    main()

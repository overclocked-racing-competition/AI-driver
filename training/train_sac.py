# SAC training driver with two-stage curriculum (stage 1 time trial -> stage 2
# multi-car). Contains transfer_stage1_to_stage2: 32->68 input-layer weight
# surgery (zero-init opponent columns, re-indexed critic action columns).
# LayerNormSACPolicy, checkpoint/resume, telemetry + TensorBoard.
# Usage: python train_sac.py --stage 1 | --stage 2 --stage1-model <ckpt.zip>

import os
import sys
import glob
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList

from config import Config
from core.torcs_env_sac import TorcsSACEnv
from core.custom_policy import LayerNormSACPolicy
from agents.bc_pretrain import build_teacher
from agents.bc_anchored_sac import BCAnchoredSAC
from core.telemetry_recorder import TelemetryRecorder
from core.callbacks import (
    TelemetryCallback,
    LapTimeCallback,
    TorcsRelaunchCallback,
    EnhancedCheckpointCallback,
    FreezeActorCallback,
)


# ==================================================================
# CLI
# ==================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAC Training for TORCS Racing AI — IBM AI Racing League"
    )
    parser.add_argument(
        "--stage", type=int, default=1, choices=[1, 2],
        help="Curriculum stage: 1=time trial, 2=multi-car (default: 1)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total training timesteps (default: from config.py)"
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Training device (default: auto)"
    )
    parser.add_argument(
        "--resume", type=str, default="auto",
        help="Path to model checkpoint .zip to resume from. Use 'auto' to find "
             "the latest automatically, or '' / 'none' to start fresh."
    )
    parser.add_argument(
        "--resume-buffer", type=str, default=None,
        help="Path to replay buffer .pkl to load (used with --resume)"
    )
    parser.add_argument(
        "--stage1-model", type=str, default=None,
        help="Path to Stage 1 model .zip for Stage 2 weight transfer"
    )
    parser.add_argument(
        "--seed-demos", type=int, default=20_000, metavar="N",
        help="Seed replay buffer with N steps of expert teacher driving before "
             "SAC training begins. Set to 0 to skip. (default: 20000)"
    )
    parser.add_argument(
        "--controller", type=str, default="v2", choices=["v1", "v2", "v3"],
        help="Teacher controller used for buffer seeding (default: v2)."
    )
    parser.add_argument(
        "--teacher-params", type=str, default=None, metavar="PATH",
        help="Path to tuned teacher params JSON (tune_teacher.py --mode export). "
             "Used for replay-buffer seeding. If omitted, uses default teacher params."
    )
    parser.add_argument(
        "--bc-pretrain", type=str, default=None, metavar="PATH",
        help="Path to BC/DAgger-pretrained .pth weights to initialize SAC actor "
             "(e.g. checkpoints/dagger_policy.pth). The network must match the actor "
             "architecture exactly (see bc_pretrain.py). Leaves critic weights random. "
             "Sets learning_starts=0 so training begins immediately."
    )
    parser.add_argument(
        "--no-telemetry", action="store_true", default=False,
        help="Disable telemetry recording"
    )
    parser.add_argument(
        "--session-name", type=str, default=None,
        help="Custom telemetry session name"
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        help="Verbosity: 0=silent, 1=info, 2=debug (default: 1)"
    )
    return parser.parse_args()


# ==================================================================
# Environment helpers
# ==================================================================

def make_env(stage: int) -> VecNormalize:
    # Create a VecNormalize-wrapped DummyVecEnv for training.
    #
    # norm_obs=False  — observations already hand-normalized in observation_utils.py.
    # norm_reward=False — CRITICAL when fine-tuning from a BC/DAgger init.
    #
    # Why norm_reward=False:
    # With norm_reward=True, VecNormalize divides rewards by the running std.
    # After seeding with 20k expert transitions (high, consistent rewards), the
    # running std is large, squashing SAC Q-values to tiny numbers (~0.001 scale).
    # The BC policy frequently outputs saturated actions (tanh ≈ ±1.0) where the
    # Jacobian correction in the SAC entropy term is large (~0.02) while the
    # squashed Q-gradient is near zero (~0.00001).  Entropy dominates → actor
    # is yanked away from full-throttle / max-steer to satisfy exploration → car
    # instantly loses speed or swerves → crash in the first lap.
    #
    # With norm_reward=False the Q-gradient stays at its natural magnitude
    # (Q ≈ 100 on straights) and the entropy term (0.05 * log_prob) is correctly
    # small relative to the Q signal.
    raw_env = DummyVecEnv([lambda: TorcsSACEnv(stage=stage)])
    env = VecNormalize(
        raw_env,
        norm_obs=False,
        norm_reward=False,
        clip_reward=10.0,
        gamma=Config.sac.gamma,
    )
    return env


def load_vecnorm(env: DummyVecEnv, vecnorm_path: str) -> VecNormalize:
    # Load saved VecNormalize statistics and wrap env.
    print(f"[VecNorm] Loading stats from: {vecnorm_path}")
    venv = VecNormalize.load(vecnorm_path, env)
    venv.training = True
    venv.norm_reward = False  # match training setting
    return venv


def _step_count_from_path(path: str) -> int:
    for part in os.path.basename(path).replace(".zip", "").split("_"):
        if part.isdigit():
            return int(part)
    return 0


def auto_find_resume(stage: int):
    # Return (model_path, buffer_path, vecnorm_path) of the latest checkpoint.
    pattern = os.path.join(Config.CHECKPOINT_DIR, f"sac_stage{stage}_*_steps.zip")
    models = glob.glob(pattern)
    if not models:
        return None, None, None

    latest = max(models, key=_step_count_from_path)
    step = _step_count_from_path(latest)
    base = os.path.join(Config.CHECKPOINT_DIR, f"sac_stage{stage}_{step}")

    buf_path     = base + "_replay_buffer.pkl"
    vecnorm_path = base + "_vecnorm.pkl"

    return (
        latest,
        buf_path if os.path.exists(buf_path) else None,
        vecnorm_path if os.path.exists(vecnorm_path) else None,
    )


# ==================================================================
# Model creation / loading
# ==================================================================

def create_model(env: VecNormalize, args) -> BCAnchoredSAC:
    # Create a fresh BCAnchoredSAC model with LayerNormSACPolicy.
    sac_cfg      = Config.sac
    training_cfg = Config.training

    lr = sac_cfg.learning_rate if args.stage == 1 else sac_cfg.learning_rate_stage2

    device = args.device if args.device != "auto" else Config.get_device()

    print(f"\n{'='*60}")
    print(f"  Creating BCAnchoredSAC Model — Stage {args.stage}")
    print(f"  Policy:       LayerNormSACPolicy")
    print(f"  Device:       {device}", end="")
    if device == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(f"  Actor arch:   {sac_cfg.pi_layers}")
    print(f"  Critic arch:  {sac_cfg.qf_layers}")
    print(f"  LR:           {lr}  |  Gamma: {sac_cfg.gamma}  |  Buffer: {sac_cfg.buffer_size:,}")
    print(f"  BC anchor:    bc_coef0={sac_cfg.bc_coef0}  decay={sac_cfg.bc_decay_steps:,} steps")
    print(f"{'='*60}\n")

    model = BCAnchoredSAC(
        policy=LayerNormSACPolicy,
        env=env,
        learning_rate=lr,
        buffer_size=sac_cfg.buffer_size,
        batch_size=sac_cfg.batch_size,
        tau=sac_cfg.tau,
        gamma=sac_cfg.gamma,
        train_freq=(sac_cfg.train_freq, "step"),
        gradient_steps=sac_cfg.gradient_steps,
        ent_coef=sac_cfg.ent_coef,
        target_entropy=sac_cfg.target_entropy,
        learning_starts=sac_cfg.learning_starts,
        policy_kwargs=sac_cfg.policy_kwargs,
        tensorboard_log=training_cfg.tensorboard_log,
        verbose=args.verbose,
        device=device,
        # BC anchor params from config
        bc_coef0=sac_cfg.bc_coef0,
        bc_decay_steps=sac_cfg.bc_decay_steps,
    )
    return model


def load_model_for_resume(model_path: str, env: VecNormalize, args) -> BCAnchoredSAC:
    # Load a checkpoint and restore the replay buffer if available.
    device = args.device if args.device != "auto" else Config.get_device()
    print(f"\n[Resume] Loading model: {model_path}")

    model = BCAnchoredSAC.load(
        model_path,
        env=env,
        device=device,
        tensorboard_log=Config.training.tensorboard_log,
        custom_objects={"policy_class": LayerNormSACPolicy},
    )

    if args.resume_buffer and os.path.exists(args.resume_buffer):
        print(f"[Resume] Loading replay buffer: {args.resume_buffer}")
        model.load_replay_buffer(args.resume_buffer)
        print(f"[Resume] Buffer loaded — {model.replay_buffer.size()} transitions")

    return model


def transfer_stage1_to_stage2(stage1_path: str, env: VecNormalize, args) -> SAC:
    # Transfer weights from a trained Stage 1 model into a fresh Stage 2 model.
    #
    # Strategy:
    # 1. Load Stage 1 weights.
    # 2. Create Stage 2 model (67-dim obs).
    # 3. Copy matching hidden weights; zero-init the 36 new opponent input connections.
    # 4. Reset replay buffer (Stage 1 experiences are incompatible).
    print(f"\n{'='*60}")
    print(f"  Stage 1 → Stage 2 Weight Transfer")
    print(f"  Source:   {stage1_path}")
    print(f"  Obs dim:  31 → 67 (+36 opponent sensors)")
    print(f"{'='*60}\n")

    device = args.device if args.device != "auto" else Config.get_device()

    stage1_model = SAC.load(
        stage1_path,
        device=device,
        custom_objects={"policy_class": LayerNormSACPolicy},
    )
    stage2_model = create_model(env, args)

    _transfer_weights(stage1_model, stage2_model, stage1_obs_dim=31, stage2_obs_dim=67)

    print("[Transfer] Weight transfer complete.")
    print("[Transfer] Replay buffer reset (Stage 1 data incompatible with Stage 2).")
    return stage2_model


def _transfer_weights(
    src_model: SAC,
    dst_model: SAC,
    stage1_obs_dim: int,
    stage2_obs_dim: int,
):
    # Copy compatible weights from Stage 1 → Stage 2.
    #
    # For the first linear layer of actor and critic (input dim changed 31→67):
    # - Copy obs weights [:, :31]
    # - Zero-init opponent weights [:, 31:67]
    # - (Critic only) copy action weights to their shifted position [:, 67:69]
    # All same-shape weights (hidden layers, output heads, LayerNorm params) copy verbatim.
    src_actor  = src_model.policy.actor.state_dict()
    dst_actor  = dst_model.policy.actor.state_dict()
    src_critic = src_model.policy.critic.state_dict()
    dst_critic = dst_model.policy.critic.state_dict()

    # ---- Actor ----
    for key in src_actor:
        if key not in dst_actor:
            continue
        src_shape = src_actor[key].shape
        dst_shape = dst_actor[key].shape

        if src_shape == dst_shape:
            dst_actor[key] = src_actor[key].clone()
        elif len(src_shape) == 2 and len(dst_shape) == 2:
            # First linear layer: only input dim differs
            dst_actor[key][:, :stage1_obs_dim] = src_actor[key][:, :stage1_obs_dim]
            if dst_shape[1] > stage1_obs_dim:
                dst_actor[key][:, stage1_obs_dim:].zero_()
        elif len(src_shape) == 1 and len(dst_shape) == 1:
            min_len = min(src_shape[0], dst_shape[0])
            dst_actor[key][:min_len] = src_actor[key][:min_len]

    dst_model.policy.actor.load_state_dict(dst_actor)

    def _transfer_critic_dict(src_dict, dst_dict):
        for key in src_dict:
            if key not in dst_dict:
                continue
            src_shape = src_dict[key].shape
            dst_shape = dst_dict[key].shape

            if src_shape == dst_shape:
                dst_dict[key] = src_dict[key].clone()
            elif len(src_shape) == 2 and len(dst_shape) == 2:
                # Critic first layer input: [obs+act] dim changes; action columns shift
                action_dim = src_shape[1] - stage1_obs_dim
                dst_dict[key][:, :stage1_obs_dim] = src_dict[key][:, :stage1_obs_dim]
                dst_dict[key][:, stage1_obs_dim:stage2_obs_dim].zero_()
                dst_dict[key][:, stage2_obs_dim:stage2_obs_dim + action_dim] = \
                    src_dict[key][:, stage1_obs_dim:stage1_obs_dim + action_dim]
            elif len(src_shape) == 1 and len(dst_shape) == 1:
                min_len = min(src_shape[0], dst_shape[0])
                dst_dict[key][:min_len] = src_dict[key][:min_len]

    _transfer_critic_dict(src_critic, dst_critic)
    dst_model.policy.critic.load_state_dict(dst_critic)

    src_target = src_model.policy.critic_target.state_dict()
    dst_target = dst_model.policy.critic_target.state_dict()
    _transfer_critic_dict(src_target, dst_target)
    dst_model.policy.critic_target.load_state_dict(dst_target)


# ==================================================================
# BC weight loading
# ==================================================================

def load_bc_weights(model: SAC, bc_path: str, verbose: int = 1):
    # Load BC-pretrained weights into the SAC actor network.
    #
    # The BC network (from bc_pretrain.py) has the same architecture as
    # LayerNormSACPolicy's actor:
    # Linear(obs_dim -> 256) -> LayerNorm -> ReLU
    # Linear(256 -> 256)     -> LayerNorm -> ReLU
    # Linear(256 -> 128)     -> LayerNorm -> ReLU
    # Linear(128 -> 2)       -> Tanh
    #
    # Strategy:
    # 1. Load BC state_dict.
    # 2. Iterate over SAC actor state dict; for each parameter, if a
    # matching BC parameter exists with the same shape, copy it.
    # 3. Leave critic weights at their random initial values.
    # 4. Set learning_starts=0 so SAC trains immediately on the seeded
    # buffer rather than waiting for 5000 random steps.
    #
    # Parameters
    # ----------
    # model : SAC
    # Freshly created SAC model (weights from LayerNormSACPolicy random init).
    # bc_path : str
    # Path to .pth file saved by bc_pretrain.py.
    # verbose : int
    # Verbosity level.
    if verbose > 0:
        print(f"\n[BC] Loading BC pretrained weights from: {bc_path}")

    bc_state = torch.load(bc_path, map_location=model.device, weights_only=True)
    actor_state = model.policy.actor.state_dict()

    matched = 0
    skipped_shape = 0
    skipped_key = 0

    # Map BCNetwork's Sequential indices (Linear/LayerNorm at 0,1,3,4,6,7 + head 9)
    # to the SAC actor's named params (latent_pi.* + mu/log_std).
    bc_to_actor = {
        0:  "latent_pi.0",   # Linear obs_dim -> 256
        1:  "latent_pi.1",   # LayerNorm 256
        3:  "latent_pi.3",   # Linear 256 -> 256
        4:  "latent_pi.4",   # LayerNorm 256
        6:  "latent_pi.6",   # Linear 256 -> 128
        7:  "latent_pi.7",   # LayerNorm 128
        9:  "mu",            # Linear 128 -> 2 (mean output head)
    }

    for bc_idx, actor_prefix in bc_to_actor.items():
        bc_key_w = f"net.{bc_idx}.weight"
        bc_key_b = f"net.{bc_idx}.bias"

        if bc_key_w in bc_state:
            w = bc_state[bc_key_w]
            actor_w_key = f"{actor_prefix}.weight"
            if actor_w_key in actor_state:
                dst = actor_state[actor_w_key]
                if dst.shape == w.shape:
                    # Exact match — direct copy
                    dst.copy_(w)
                    matched += 1
                elif (dst.dim() == 2 and w.dim() == 2
                      and dst.shape[0] == w.shape[0]
                      and dst.shape[1] > w.shape[1]):
                    # Input-dim mismatch (e.g. checkpoint=31-dim, actor=32-dim).
                    # This happens when a .pth was trained before prev_steer was added.
                    # Copy the overlapping columns; zero-pad the new column(s).
                    # The new column corresponds to prev_steer (always 0 at reset),
                    # so a zero weight is the correct mathematical initialization.
                    cols = w.shape[1]
                    dst[:, :cols].copy_(w)
                    dst[:, cols:].zero_()
                    matched += 1
                    if verbose > 0:
                        print(f"[BC] Input-dim mismatch: padded {actor_prefix}.weight "
                              f"{list(w.shape)} → {list(dst.shape)} (zero-padded {dst.shape[1]-cols} cols)")
                else:
                    skipped_shape += 1
            else:
                skipped_key += 1

            if bc_key_b in bc_state:
                b = bc_state[bc_key_b]
                actor_b_key = f"{actor_prefix}.bias"
                if actor_b_key in actor_state and actor_state[actor_b_key].shape == b.shape:
                    actor_state[actor_b_key].copy_(b)
                    matched += 1
                else:
                    skipped_shape += 1
        else:
            skipped_key += 1

    # Note: mu.weight and mu.bias are handled by the loop above (bc_idx=9 → "mu").
    # The loop uses bc_key_b = "net.9.bias" which correctly matches bc_pretrain.py's
    # key naming, fixing the original bug where "mu.bias" was checked against bc_state.

    # Commit all weight changes to the actual model.
    # CRITICAL: actor_state is a view of the model's tensors (copy_(w) above writes
    # directly into the model's storage), but load_state_dict ensures all buffers
    # (including LayerNorm running stats) are also updated correctly.
    model.policy.actor.load_state_dict(actor_state)

    # Initialize log_std to give a small, controlled exploration noise.
    # log_std_init = -3.0  →  std = exp(-3.0) ≈ 0.05 (5% noise).
    # This is large enough for SAC to distinguish Q-values and explore,
    # but small enough not to overpower the BC mean at racing speeds.
    # (S3-a used -5.0 / 0.67% — too small for real SAC exploration.)
    log_std_init = Config.sac.log_std_init
    with torch.no_grad():
        model.policy.actor.log_std.weight.fill_(0.0)
        model.policy.actor.log_std.bias.fill_(log_std_init)

    # Set learning_starts to 0 so training begins immediately on the seeded buffer
    model.learning_starts = 0

    if verbose > 0:
        print(f"[BC] Copied {matched} parameter tensors into SAC actor "
              f"({skipped_shape} shape mismatches, {skipped_key} missing keys).")
        print(f"[BC] log_std initialized: bias={log_std_init}  →  std≈{np.exp(log_std_init):.4f}")
        print(f"[BC] SAC actor initialized from BC policy. Critic remains random.")
        print(f"[BC] learning_starts set to 0.")


# ==================================================================
# Demo buffer seeding
# ==================================================================

def seed_replay_buffer(model: BCAnchoredSAC, env: VecNormalize, n_steps: int, teacher, verbose: int = 1):
    # Fill the SAC replay buffer with transitions produced by the tuned expert
    # teacher controller before SAC training begins.
    #
    # Why: this puts real lap-completing trajectories (with the +500 lap bonus) into
    # the buffer so the critic immediately sees what "good" looks like. Combined with
    # BC/DAgger actor initialization, SAC starts from a competent policy and refines it
    # rather than exploring from scratch.
    #
    # `teacher` is a pre-built controller instance (v2 by default), created via
    # bc_pretrain.build_teacher(). It is reset at every episode boundary.
    #
    # Implementation notes:
    # - We step the VecNormalize-wrapped env so reward normalisation stats
    # update consistently with what SB3 does during training.
    # - norm_obs=False so observations pass through unchanged.
    # - VecEnv auto-resets on done; terminal observation is in
    # infos[0]['terminal_observation'] — we use that as next_obs when done.
    # - After seeding, model.learning_starts is set to 0 so training starts
    # immediately on the rich demo data rather than waiting for 5000 more
    # random steps.
    #
    # Parameters
    # ----------
    # model : SAC
    # Freshly created SAC model (replay buffer is empty).
    # env : VecNormalize
    # The training environment (DummyVecEnv wrapped in VecNormalize).
    # n_steps : int
    # Number of demo transitions to collect.
    # verbose : int
    # Verbosity level.
    if n_steps <= 0:
        return

    print(f"\n[Seed] Seeding replay buffer with {n_steps:,} expert teacher transitions...")
    print(f"[Seed] Expert driver: tuned TeacherController")

    # Reset env; get initial obs and the underlying raw_obs dict
    obs = env.reset()
    # Access the raw TorcsSACEnv through VecNormalize → DummyVecEnv → env list
    raw_env: TorcsSACEnv = env.venv.envs[0]
    teacher.reset()

    laps = 0
    max_dist = 0.0
    t0 = time.time()

    # Accumulate (obs, action) pairs for the BCAnchoredSAC demo buffer.
    # These are the expert observations+actions; the BC anchor will use them to
    # regularize the SAC actor toward the expert throughout fine-tuning.
    demo_obs_list:  list = []
    demo_acts_list: list = []

    for step in range(n_steps):
        raw_obs = raw_env.get_raw_obs()
        action  = teacher.act(raw_obs)                 # shape (2,)
        action_batch = action.reshape(1, -1)           # (1, 2) for VecEnv

        # Collect demo data BEFORE stepping (obs is the current state)
        demo_obs_list.append(obs[0].copy())   # (obs_dim,)
        demo_acts_list.append(action.copy())  # (2,)

        next_obs, rewards, dones, infos = env.step(action_batch)
        # With norm_reward=False, rewards == original reward (no normalization applied).
        # Use rewards directly to match what SB3's collect_rollouts stores online.
        reward_for_buffer = rewards

        # VecEnv auto-resets when done; terminal obs is in info
        if dones[0]:
            terminal_obs = infos[0].get("terminal_observation", next_obs[0])
            effective_next_obs = np.array([terminal_obs], dtype=np.float32)
            teacher.reset()  # new episode begins next step — reset teacher state
        else:
            effective_next_obs = next_obs

        model.replay_buffer.add(
            obs=obs,
            next_obs=effective_next_obs,
            action=action_batch,
            reward=reward_for_buffer,
            done=dones,
            infos=infos,
        )

        obs = next_obs

        # Track quality
        dist = float(infos[0].get("distRaced", 0.0)) if infos else 0.0
        max_dist = max(max_dist, dist)
        if infos and infos[0].get("lap_completed", False):
            laps += 1

        if verbose > 0 and (step + 1) % 2000 == 0:
            elapsed = time.time() - t0
            fps = (step + 1) / elapsed
            print(f"[Seed] {step+1:6,}/{n_steps:,} | "
                  f"buf: {model.replay_buffer.size():,} | "
                  f"max_dist: {max_dist:.0f}m | laps: {laps} | "
                  f"fps: {fps:.0f}")

    elapsed = time.time() - t0
    print(f"\n[Seed] Complete in {elapsed:.0f}s:")
    print(f"[Seed]   Transitions: {model.replay_buffer.size():,}")
    print(f"[Seed]   Max distance: {max_dist:.0f} m")
    print(f"[Seed]   Laps completed: {laps}")
    if laps == 0:
        print(f"[Seed] WARNING: No laps completed during seeding — "
              f"check the teacher params / env if max_dist is very low.")
    else:
        print(f"[Seed] Lap-completion bonus transitions are in the buffer.")

    # Pass demo data to the BC anchor so it can regularize the actor.
    if isinstance(model, BCAnchoredSAC) and demo_obs_list:
        demo_obs  = np.stack(demo_obs_list,  axis=0)   # (n_steps, obs_dim)
        demo_acts = np.stack(demo_acts_list, axis=0)   # (n_steps, 2)
        model.add_demo_data(demo_obs, demo_acts)

    # Start training immediately — don't wait for learning_starts more random steps
    model.learning_starts = 0


# ==================================================================
# Callbacks
# ==================================================================

def build_callbacks(args, recorder: TelemetryRecorder) -> CallbackList:
    training_cfg = Config.training
    callbacks = []

    # Freeze actor at start if BC weights were loaded.
    # The FreezeActorCallback prevents the random critic from corrupting the
    # BC-initialized actor during the first `freeze_steps` environment steps.
    # After unfreeze, the BCAnchoredSAC's decaying BC loss provides ongoing protection.
    if args.bc_pretrain and os.path.exists(args.bc_pretrain):
        callbacks.append(FreezeActorCallback(
            freeze_steps=Config.sac.freeze_steps,
            verbose=args.verbose,
        ))

    if not args.no_telemetry:
        callbacks.append(TelemetryCallback(
            recorder=recorder,
            car_freq=training_cfg.car_telemetry_every_n_steps,
            neuron_freq=training_cfg.neuron_telemetry_every_n_steps,
            verbose=args.verbose,
        ))

    callbacks.append(LapTimeCallback(verbose=args.verbose))
    callbacks.append(TorcsRelaunchCallback(verbose=args.verbose))

    save_freq = (
        training_cfg.stage1_checkpoint_freq
        if args.stage == 1
        else training_cfg.stage2_checkpoint_freq
    )
    callbacks.append(EnhancedCheckpointCallback(
        save_freq=save_freq,
        save_path=Config.CHECKPOINT_DIR,
        name_prefix=f"sac_stage{args.stage}",
        save_replay_buffer=True,
        save_vecnorm=True,
        verbose=args.verbose,
    ))

    return CallbackList(callbacks)


# ==================================================================
# Main
# ==================================================================

def main():
    args = parse_args()

    # CPU thread tuning (no effect on GPU runs). Small nets often train fastest
    # with a low thread count; tune Config.sac.cpu_threads for your machine.
    resolved_device = args.device if args.device != "auto" else Config.get_device()
    if resolved_device == "cpu":
        torch.set_num_threads(Config.sac.cpu_threads)
        print(f"[CPU] torch.set_num_threads({Config.sac.cpu_threads})")

    print(Config.summary())

    total_timesteps = args.timesteps or (
        Config.training.stage1_total_timesteps
        if args.stage == 1
        else Config.training.stage2_total_timesteps
    )

    # ---- Resolve resume path ----
    resume_path     = None
    vecnorm_path    = None
    reset_timesteps = True

    if args.resume.lower() not in ("", "none", "false"):
        if args.resume == "auto":
            resume_path, buf_path, vecnorm_path = auto_find_resume(args.stage)
            if resume_path:
                print(f"\n[*] Auto-resume: found checkpoint at {_step_count_from_path(resume_path):,} steps")
                if not args.resume_buffer:
                    args.resume_buffer = buf_path
                reset_timesteps = False
            else:
                print(f"\n[*] Auto-resume: no checkpoints found — starting fresh.")
        else:
            resume_path = args.resume
            reset_timesteps = False

    # ---- Create environment ----
    print(f"\n[Init] Creating TORCS environment — Stage {args.stage}...")
    if vecnorm_path:
        raw_env = DummyVecEnv([lambda: TorcsSACEnv(stage=args.stage)])
        env = load_vecnorm(raw_env, vecnorm_path)
    else:
        env = make_env(args.stage)

    # ---- Create / load model ----
    if resume_path:
        model = load_model_for_resume(resume_path, env, args)
    elif args.stage == 2 and args.stage1_model:
        model = transfer_stage1_to_stage2(args.stage1_model, env, args)
    else:
        model = create_model(env, args)

    # ---- Load BC pretrained weights into SAC actor (if requested) ----
    # Must happen BEFORE replay buffer seeding so the demo transitions
    # are consistent with the BC-initialized policy.
    if args.bc_pretrain and not resume_path:
        load_bc_weights(model, args.bc_pretrain, verbose=args.verbose)
        # BC initializes the actor to drive like the demo; start learning immediately
        model.learning_starts = 0

    # ---- Telemetry ----
    session_name = args.session_name or f"stage{args.stage}_{time.strftime('%Y%m%d_%H%M%S')}"
    recorder = TelemetryRecorder(
        session_name=session_name,
        enabled=not args.no_telemetry,
    )

    callbacks = build_callbacks(args, recorder)

    # ---- Seed replay buffer with expert teacher transitions (if requested) ----
    # Skipped on resume (buffer already contains real experiences).
    if args.seed_demos > 0 and not resume_path:
        teacher = build_teacher(args.controller, args.teacher_params)
        seed_replay_buffer(model, env, n_steps=args.seed_demos,
                           teacher=teacher, verbose=args.verbose)
    elif args.seed_demos > 0 and resume_path:
        print(f"[Seed] Skipping teacher seeding (resuming from checkpoint — "
              f"buffer already populated).")

    # ---- Train ----
    device = args.device if args.device != "auto" else Config.get_device()
    print(f"\n{'='*60}")
    print(f"  Starting SAC Training — Stage {args.stage}")
    print(f"  Timesteps:  {total_timesteps:,}")
    print(f"  Device:     {device}")
    if device == "cuda":
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Checkpoint: {Config.CHECKPOINT_DIR}")
    print(f"  Logs:       {Config.LOG_DIR}")
    print(f"{'='*60}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            tb_log_name=f"sac_stage{args.stage}",
            reset_num_timesteps=reset_timesteps,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")
    finally:
        final_path = os.path.join(Config.CHECKPOINT_DIR, f"sac_stage{args.stage}_final")
        model.save(final_path)
        print(f"\n[Save] Model → {final_path}.zip")

        buf_path = os.path.join(Config.CHECKPOINT_DIR, f"sac_stage{args.stage}_final_replay_buffer")
        model.save_replay_buffer(buf_path)
        print(f"[Save] Replay buffer → {buf_path}.pkl")

        vn_path = os.path.join(Config.CHECKPOINT_DIR, f"sac_stage{args.stage}_final_vecnorm.pkl")
        env.save(vn_path)
        print(f"[Save] VecNormalize stats → {vn_path}")

        recorder.close()
        env.close()
        print("\n[Done] Training complete.")


if __name__ == "__main__":
    main()

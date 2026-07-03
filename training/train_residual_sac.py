# Residual SAC over a frozen base policy (collapse-proof):
#   final_action = clip(base(obs) + delta * residual(obs), -1, 1)
# residual=0 reproduces the base lap exactly -> structural performance floor.
# Pipeline: seed replay with residual=0 rollouts -> critic warmup (frozen actor)
# -> SAC learns bounded corrections. Periodic floor checks + --eval mode.
# Usage: python train_residual_sac.py --dagger-weights <base.pth> --resume none
#        python train_residual_sac.py --eval checkpoints/residual_sac_<N>_steps.zip

import os
import sys
import glob
import argparse
import time
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

from config import Config
from training.residual_env import ResidualTorcsEnv
from core.custom_policy import LayerNormSACPolicy
from agents.bc_anchored_sac import BCAnchoredSAC
from core.telemetry_recorder import TelemetryRecorder
from core.callbacks import (
    TelemetryCallback,
    LapTimeCallback,
    TorcsRelaunchCallback,
    EnhancedCheckpointCallback,
    FreezeActorCallback,
)

CHECKPOINT_PREFIX = "residual_sac"


# ==================================================================
# Environment
# ==================================================================

def make_env(dagger_weights: str = None) -> VecNormalize:
    # Create VecNormalize-wrapped ResidualTorcsEnv.
    # norm_reward=False — see train_sac.py for the full explanation.
    cfg = Config.residual
    dw = dagger_weights or cfg.dagger_weights

    raw_env = DummyVecEnv([lambda: ResidualTorcsEnv(dagger_weights=dw)])
    return VecNormalize(
        raw_env,
        norm_obs=False,
        norm_reward=False,
        clip_reward=10.0,
        gamma=cfg.gamma,
    )


# ==================================================================
# Model
# ==================================================================

def create_model(env: VecNormalize, args) -> BCAnchoredSAC:
    # Create a fresh BCAnchoredSAC model with the residual RL hyperparameters.
    cfg    = Config.residual
    lr     = args.lr or cfg.learning_rate
    device = args.device if args.device != "auto" else Config.get_device()

    print(f"\n{'='*60}")
    print(f"  Creating Residual SAC Model (BCAnchoredSAC)")
    print(f"  Policy:       LayerNormSACPolicy")
    print(f"  Device:       {device}")
    print(f"  delta:        steer={cfg.delta_steer} (±{cfg.delta_steer*100:.0f}%), accel={cfg.delta_accel} (±{cfg.delta_accel*100:.0f}%)")
    print(f"  LR:           {lr}  |  ent_coef: {cfg.ent_coef}")
    print(f"  Buffer:       {cfg.buffer_size:,}")
    print(f"  BC Anchor:    bc_coef0={cfg.bc_coef0}, decay={cfg.bc_decay_steps}")
    print(f"{'='*60}\n")

    model = BCAnchoredSAC(
        policy=LayerNormSACPolicy,
        env=env,
        learning_rate=lr,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        ent_coef=cfg.ent_coef,
        learning_starts=cfg.freeze_steps,   # handled by FreezeActorCallback too
        use_sde=cfg.use_sde,                # smooth exploration (no high-freq jitter)
        sde_sample_freq=cfg.sde_sample_freq,
        bc_coef0=cfg.bc_coef0,
        bc_decay_steps=cfg.bc_decay_steps,
        policy_kwargs=dict(
            net_arch=dict(pi=Config.sac.pi_layers, qf=Config.sac.qf_layers),
            activation_fn=torch.nn.ReLU,
            use_sde=cfg.use_sde,
        ),
        tensorboard_log=Config.LOG_DIR,
        verbose=args.verbose,
        device=device,
    )

    # --- Initialize the residual actor for GENTLE exploration starting at DAgger ---
    # Zero mu → initial residual = 0 = exact DAgger. Set log_std low → gentle exploration.
    # Structures differ:
    #   standard SAC: mu = Linear,                 log_std = Linear
    #   gSDE:         mu = Sequential(Linear,...),  log_std = Parameter(features, act_dim)
    with torch.no_grad():
        if cfg.zero_init_mu:
            mu = model.policy.actor.mu
            if isinstance(mu, torch.nn.Linear):
                mu.weight.zero_(); mu.bias.zero_()
            else:  # Sequential (gSDE) — zero the Linear(s) inside so mean output = 0
                for sub in mu.modules():
                    if isinstance(sub, torch.nn.Linear):
                        sub.weight.zero_(); sub.bias.zero_()

        log_std = model.policy.actor.log_std
        if isinstance(log_std, torch.nn.Linear):           # standard SAC
            log_std.weight.zero_()
            log_std.bias.fill_(cfg.log_std_init)
        elif isinstance(log_std, torch.nn.Parameter):      # gSDE exploration matrix
            log_std.data.fill_(cfg.log_std_init)

        # Re-sync gSDE exploration matrix to the new log_std before any rollout
        if cfg.use_sde and hasattr(model.actor, "reset_noise"):
            model.actor.reset_noise()

    print(f"  Residual actor: mu zeroed (start = DAgger), "
          f"log_std={cfg.log_std_init}, gSDE={'ON' if cfg.use_sde else 'OFF'} "
          f"(sde_sample_freq={cfg.sde_sample_freq})")

    return model


def _step_count_from_path(path: str) -> int:
    for part in os.path.basename(path).replace(".zip", "").split("_"):
        if part.isdigit():
            return int(part)
    return 0


def auto_find_resume():
    # Return (model_path, buffer_path, vecnorm_path) of the latest residual checkpoint.
    pattern = os.path.join(Config.CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_*_steps.zip")
    models  = glob.glob(pattern)
    if not models:
        return None, None, None

    latest = max(models, key=_step_count_from_path)
    step   = _step_count_from_path(latest)
    base   = os.path.join(Config.CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_{step}")

    buf_path     = base + "_replay_buffer.pkl"
    vecnorm_path = base + "_vecnorm.pkl"
    return (
        latest,
        buf_path     if os.path.exists(buf_path)     else None,
        vecnorm_path if os.path.exists(vecnorm_path) else None,
    )


# ==================================================================
# Seeding (DAgger baseline — residual = 0)
# ==================================================================

def seed_replay_buffer(model: BCAnchoredSAC, env: VecNormalize, n_steps: int, verbose: int = 1):
    # Seed the replay buffer with the DAgger BASELINE (residual = 0).
    #
    # Why residual=0 seeding (not teacher seeding):
    # - The DAgger policy now completes full laps at ~1:47.
    # - Seeding with residual=0 means the env applies dagger(obs) as the
    # final action, which is perfectly self-consistent in the buffer:
    # (obs, residual=0, reward_from_dagger, next_obs).
    # - The critic immediately learns Q(s, residual=0) ~ "following DAgger
    # gives good rewards including the +500 lap bonus."
    # - SAC then explores small non-zero residuals and learns to beat it.
    # - No teacher needed, no residual-computation complexity.
    if n_steps <= 0:
        return

    print(f"\n[Seed] Seeding {n_steps:,} steps with DAgger baseline (residual=0)...")
    print(f"[Seed] The DAgger policy laps at ~1:47 — buffer will contain full-lap data.")

    obs = env.reset()
    laps = 0
    max_dist = 0.0
    t0 = time.time()

    zero_residual = np.zeros((1, 2), dtype=np.float32)
    demo_obs_list = []
    demo_acts_list = []

    for step in range(n_steps):
        next_obs, rewards, dones, infos = env.step(zero_residual)

        demo_obs_list.append(obs[0].copy())
        demo_acts_list.append(zero_residual[0].copy())

        if dones[0]:
            terminal_obs = infos[0].get("terminal_observation", next_obs[0])
            effective_next = np.array([terminal_obs], dtype=np.float32)
        else:
            effective_next = next_obs

        model.replay_buffer.add(
            obs=obs,
            next_obs=effective_next,
            action=zero_residual,
            reward=rewards,
            done=dones,
            infos=infos,
        )
        obs = next_obs

        dist = float(infos[0].get("distRaced", 0.0)) if infos else 0.0
        max_dist = max(max_dist, dist)
        if infos and infos[0].get("lap_completed", False):
            laps += 1

        if verbose and (step + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"[Seed] {step+1:6,}/{n_steps:,} | buf: {model.replay_buffer.size():,} | "
                  f"max_dist: {max_dist:.0f}m | laps: {laps} | fps: {(step+1)/elapsed:.0f}")

    elapsed = time.time() - t0
    print(f"\n[Seed] Complete in {elapsed:.0f}s  |  buf: {model.replay_buffer.size():,}  |  "
          f"max_dist: {max_dist:.0f}m  |  laps: {laps}")
    if laps == 0:
        print(f"[Seed] WARNING: DAgger baseline completed no laps during seeding. "
              f"Check dagger_policy_v2.pth — it should lap at ~1:47.")
    else:
        print(f"[Seed] Lap-completion bonus (+500) transitions are in the buffer.")

    if hasattr(model, "add_demo_data") and demo_obs_list:
        model.add_demo_data(np.array(demo_obs_list), np.array(demo_acts_list))
        print(f"[Seed] Injected {len(demo_obs_list)} transitions into BC Anchor demo buffer.")

    model.learning_starts = 0


# ==================================================================
# Floor check — verifies the DAgger baseline is preserved
# ==================================================================

def evaluate_learned(model: BCAnchoredSAC, args, n_episodes: int = 2, step_label: str = ""):
    # Evaluate the LEARNED policy deterministically — this is what tells us whether
    # SAC is actually improving on DAgger.
    #
    # Runs model.predict(deterministic=True) → residual → final = dagger + delta*residual.
    # Reports lap time vs the DAgger baseline (106.96s). This is the REAL progress metric
    # (verify_floor only checks residual=0, which is structurally constant = DAgger).
    cfg = Config.residual
    dw  = args.dagger_weights or cfg.dagger_weights

    raw = DummyVecEnv([lambda: ResidualTorcsEnv(dagger_weights=dw)])
    env = VecNormalize(raw, norm_obs=False, norm_reward=False, clip_reward=10.0)
    env.training = False

    max_dists, lap_times = [], []
    print(f"\n[EvalLearned{step_label}] {n_episodes} episodes, deterministic LEARNED policy...")
    obs = env.reset()

    for ep in range(n_episodes):
        ep_max, last_lap = 0.0, None
        for _ in range(Config.torcs.max_steps_per_episode):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            ep_max = max(ep_max, float(infos[0].get("distRaced", 0.0)))
            if infos[0].get("lap_completed", False):
                llt = float(infos[0].get("lastLapTime", 0.0))
                if llt > 0 and llt != last_lap:
                    last_lap = llt
                    lap_times.append(llt)
            if dones[0]:
                break
        max_dists.append(ep_max)
        print(f"  Ep {ep+1}: dist={ep_max:.0f}m" + (f"  lap={last_lap:.2f}s" if last_lap else "  no lap"))

    env.close()
    best = min(lap_times) if lap_times else float("inf")
    avg_dist = float(np.mean(max_dists))
    print(f"[EvalLearned] avg_dist={avg_dist:.0f}m  "
          f"best_lap={'%.2f' % best if lap_times else 'no lap'}  "
          f"(DAgger baseline = 106.96s)")
    if lap_times and best < 106.96:
        print(f"[EvalLearned] *** SAC IMPROVED on DAgger by {106.96 - best:.2f}s! ***")
    elif lap_times:
        print(f"[EvalLearned] Learned policy laps but not faster yet ({best:.2f}s vs 106.96s).")
    return avg_dist, best


def verify_floor(model: SAC, args, step_label: str = ""):
    # Run a short deterministic evaluation with residual=0 (DAgger floor check).
    # If max_dist < 1000 m, the floor has been broken — training should stop.
    # Called before training and periodically during training.
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    cfg  = Config.residual
    dw   = args.dagger_weights or cfg.dagger_weights
    dev  = args.device if args.device != "auto" else Config.get_device()

    # Fresh env — DummyVecEnv wrapping ResidualTorcsEnv
    raw = DummyVecEnv([lambda: ResidualTorcsEnv(dagger_weights=dw)])
    env = VecNormalize(raw, norm_obs=False, norm_reward=False, clip_reward=10.0)

    max_dists = []
    lap_times = []
    n_ep = 2

    print(f"\n[FloorCheck{step_label}] Running {n_ep} episodes with residual=0 (DAgger baseline)...")
    obs = env.reset()
    zero = np.zeros((1, 2), dtype=np.float32)

    for ep in range(n_ep):
        ep_max = 0.0
        last_lap = None
        for _ in range(Config.torcs.max_steps_per_episode):
            obs, _, dones, infos = env.step(zero)
            dist = float(infos[0].get("distRaced", 0.0))
            ep_max = max(ep_max, dist)
            if infos[0].get("lap_completed", False):
                llt = float(infos[0].get("lastLapTime", 0.0))
                if llt > 0 and llt != last_lap:
                    last_lap = llt
                    lap_times.append(llt)
            if dones[0]:
                break
        max_dists.append(ep_max)
        print(f"  Ep {ep+1}: dist={ep_max:.0f}m" + (f"  lap={last_lap:.2f}s" if last_lap else ""))

    env.close()

    avg_dist = np.mean(max_dists)
    best_lap = min(lap_times) if lap_times else float("inf")
    floor_ok = avg_dist >= 1000.0  # full lap is 3602m; 1000m is a generous floor

    print(f"[FloorCheck] avg_dist={avg_dist:.0f}m  "
          f"best_lap={'%.2f' % best_lap if lap_times else 'no lap'}")
    if floor_ok:
        print(f"[FloorCheck] FLOOR INTACT — DAgger baseline preserved.")
    else:
        print(f"[FloorCheck] *** FLOOR BROKEN *** avg_dist {avg_dist:.0f}m < 1000m. "
              f"The DAgger base has been corrupted. Stop training and investigate.")
    return floor_ok, avg_dist, best_lap


# ==================================================================
# Floor-check callback
# ==================================================================

class ResidualFloorCheckCallback:
    # Minimal inline floor-check (not a full SB3 callback — avoids env clash).
    # Called from the training loop every eval_freq steps.
    def __init__(self, model, args, eval_freq: int):
        self.model     = model
        self.args      = args
        self.eval_freq = eval_freq
        self._last_check = 0

    def check_if_due(self):
        if self.model.num_timesteps - self._last_check >= self.eval_freq:
            self._last_check = self.model.num_timesteps
            ok, dist, lap = verify_floor(self.model, self.args,
                                         step_label=f"@{self.model.num_timesteps:,}")
            return ok, dist, lap
        return True, None, None


# ==================================================================
# Entropy decay callback
# ==================================================================

class EntropyDecayCallback(BaseCallback):
    # Linearly decays ent_coef from its initial value to ent_coef_final over
    # [decay_start, decay_end] timesteps, then holds at ent_coef_final.
    #
    # Why this helps: Optuna finds ent_coef ~0.04-0.05 is good for EXPLORATION
    # (finds fast lines in first 100k steps), but that same high value keeps the
    # actor exploring PAST the good lines and destroys the policy. This callback
    # lets SAC explore freely early, then converges to a stable exploitation policy.
    # The BestCheckpointEvalCallback captures whichever peak is fastest.

    def __init__(self, ent_coef_init: float, ent_coef_final: float,
                 decay_start: int, decay_end: int, verbose: int = 1):
        super().__init__(verbose)
        self.ent_coef_init  = ent_coef_init
        self.ent_coef_final = ent_coef_final
        self.decay_start    = decay_start
        self.decay_end      = decay_end
        self._logged_start  = False

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t < self.decay_start:
            return True

        if not self._logged_start:
            self._logged_start = True
            if self.verbose:
                print(f"\n[EntDecay] Entropy decay started at step {t:,}: "
                      f"{self.ent_coef_init:.4f} -> {self.ent_coef_final:.4f} "
                      f"over {self.decay_end - self.decay_start:,} steps")

        if t >= self.decay_end:
            new_val = self.ent_coef_final
        else:
            frac = (t - self.decay_start) / (self.decay_end - self.decay_start)
            new_val = self.ent_coef_init + frac * (self.ent_coef_final - self.ent_coef_init)

        self.model.ent_coef_tensor = torch.tensor(
            new_val, device=self.model.device, dtype=torch.float32
        )
        return True


# ==================================================================
# Best-checkpoint callback (LOCAL training only — not used by Optuna)
# ==================================================================

class BestCheckpointEvalCallback(BaseCallback):
    # Periodically evaluates the DETERMINISTIC learned policy and saves a separate
    # `<prefix>_best.zip` whenever it improves. Solves the core problem we hit: the
    # deterministic policy peaks EARLY then degrades, and plain periodic checkpoints
    # don't tell you which one is good — the fast policy gets trained away and lost.
    #
    # Selection rule:
    # - Once any eval completes a lap, only FASTER laps overwrite the best
    # (a non-lapping eval can never replace a lapping one).
    # - Before the first lap, the farthest-distance eval is kept (progress signal).
    #
    # Single continuous learn() is preserved (no telemetry-CSV close bug): the eval
    # runs inside _on_step on its own short-lived env, exactly like the Optuna path.

    def __init__(self, args, eval_freq: int = 20_000, n_eval_episodes: int = 2,
                 save_path: str = None, verbose: int = 1):
        super().__init__(verbose)
        self.args            = args
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path       = save_path or os.path.join(
            Config.CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_best")
        self.best_lap   = float("inf")
        self.best_dist  = 0.0
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps

        avg_dist, best_lap = evaluate_learned(
            self.model, self.args, n_episodes=self.n_eval_episodes,
            step_label=f"@{self.num_timesteps // 1000}k",
        )

        improved = False
        if best_lap < float("inf"):
            # We have a lap this eval — only a FASTER lap counts as an improvement.
            if best_lap < self.best_lap:
                self.best_lap = best_lap
                improved = True
        elif self.best_lap == float("inf"):
            # Never lapped yet — track farthest distance as the progress signal.
            if avg_dist > self.best_dist:
                self.best_dist = avg_dist
                improved = True

        if improved:
            self.model.save(self.save_path)
            tag = (f"{self.best_lap:.2f}s lap" if self.best_lap < float("inf")
                   else f"{self.best_dist:.0f}m (no lap yet)")
            print(f"[BestCkpt] New best: {tag}  ->  saved {self.save_path}.zip "
                  f"(@{self.num_timesteps:,} steps)")
        else:
            cur = (f"{best_lap:.2f}s" if best_lap < float("inf") else f"{avg_dist:.0f}m")
            held = (f"{self.best_lap:.2f}s" if self.best_lap < float("inf")
                    else f"{self.best_dist:.0f}m")
            print(f"[BestCkpt] No improvement (this={cur}, best held={held}) "
                  f"— best checkpoint preserved.")
        return True


# ==================================================================
# Callbacks
# ==================================================================

def build_callbacks(args, recorder: TelemetryRecorder) -> CallbackList:
    training_cfg = Config.training
    residual_cfg = Config.residual
    cbs = []

    # Always freeze the actor at training start so the critic warms up first
    cbs.append(FreezeActorCallback(
        freeze_steps=residual_cfg.freeze_steps, verbose=args.verbose
    ))

    if not args.no_telemetry:
        cbs.append(TelemetryCallback(
            recorder=recorder,
            car_freq=training_cfg.car_telemetry_every_n_steps,
            neuron_freq=training_cfg.neuron_telemetry_every_n_steps,
            verbose=args.verbose,
        ))

    cbs.append(LapTimeCallback(verbose=args.verbose))
    cbs.append(TorcsRelaunchCallback(verbose=args.verbose))
    cbs.append(EnhancedCheckpointCallback(
        save_freq=residual_cfg.checkpoint_freq,
        save_path=Config.CHECKPOINT_DIR,
        name_prefix=CHECKPOINT_PREFIX,
        save_replay_buffer=True,
        save_vecnorm=True,
        verbose=args.verbose,
    ))

    # Best-checkpoint capture: deterministic eval every eval_freq steps, keep the
    # fastest-lapping policy in <prefix>_best.zip (the peak is usually early — this
    # stops us from training past it and losing it).
    if not getattr(args, "no_best_eval", False):
        cbs.append(BestCheckpointEvalCallback(
            args,
            eval_freq=getattr(args, "eval_freq", 20_000),
            n_eval_episodes=2,
            verbose=args.verbose,
        ))

    # Entropy decay: explore freely early, stabilise once good lines are found.
    # Activated only when --ent-coef-final is set (no decay = existing behaviour).
    if getattr(args, "ent_coef_final", None) is not None:
        ent_init   = residual_cfg.ent_coef   # already overridden by --ent-coef if passed
        d_start    = getattr(args, "ent_coef_decay_start", 50_000)
        d_steps    = getattr(args, "ent_coef_decay_steps", 300_000)
        cbs.append(EntropyDecayCallback(
            ent_coef_init=ent_init,
            ent_coef_final=args.ent_coef_final,
            decay_start=d_start,
            decay_end=d_start + d_steps,
            verbose=args.verbose,
        ))

    return CallbackList(cbs)


# ==================================================================
# CLI
# ==================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Residual RL SAC training over frozen DAgger")
    p.add_argument("--dagger-weights", type=str,   default=None,
                   help="Path to frozen DAgger .pth (default: Config.residual.dagger_weights)")
    p.add_argument("--timesteps",      type=int,   default=None,
                   help="Total training timesteps (default: Config.residual.total_timesteps)")
    p.add_argument("--lr",             type=float, default=None,
                   help="Learning rate (default: Config.residual.learning_rate)")
    p.add_argument("--ent-coef",       type=float, default=None,
                   help="Entropy coefficient (default: Config.residual.ent_coef=0.02)")
    p.add_argument("--log-std-init",   type=float, default=None,
                   help="gSDE log_std init (default: Config.residual.log_std_init=-3.5)")
    p.add_argument("--gamma",          type=float, default=None,
                   help="Discount factor (default: Config.residual.gamma=0.99)")
    p.add_argument("--tau",            type=float, default=None,
                   help="Soft target update (default: Config.residual.tau=0.005)")
    p.add_argument("--seed-demos",     type=int,   default=None,
                   help="DAgger baseline seeding steps (default: Config.residual.seed_steps)")
    p.add_argument("--device",         type=str,   default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--resume",         type=str,   default="auto",
                   help="Checkpoint .zip to resume from ('auto' finds latest, '' for fresh start)")
    p.add_argument("--eval",           type=str,   default=None,
                   help="Evaluate a residual_sac checkpoint's LEARNED policy (deterministic) and exit. "
                        "e.g. --eval checkpoints/residual_sac_final.zip")
    p.add_argument("--skip-floor-check", action="store_true", default=False,
                   help="Skip the pre-training floor verification (not recommended)")
    p.add_argument("--eval-freq",      type=int,   default=20_000,
                   help="Steps between deterministic best-checkpoint evals (default 20000)")
    p.add_argument("--no-best-eval",   action="store_true", default=True,
                   help="Disable the periodic best-checkpoint eval/save")
    p.add_argument("--ent-coef-final",       type=float, default=None,
                   help="Final ent_coef after decay (None=no decay; try 0.002 to stabilize policy)")
    p.add_argument("--ent-coef-decay-start", type=int,   default=50_000,
                   help="Step at which entropy decay begins (default 50k)")
    p.add_argument("--ent-coef-decay-steps", type=int,   default=300_000,
                   help="Steps over which ent_coef decays to --ent-coef-final (default 300k)")
    p.add_argument("--no-telemetry",   action="store_true", default=False)
    p.add_argument("--session-name",   type=str,   default=None)
    p.add_argument("--verbose",        type=int,   default=1)
    return p.parse_args()


# ==================================================================
# Main
# ==================================================================

def main():
    args = parse_args()
    cfg  = Config.residual

    # ---- Apply CLI overrides to Config.residual (before make_env / create_model read it) ----
    # This allows reproducing any Optuna trial exactly, e.g.:
    #   python train_residual_sac.py --delta 0.244 --lr 1.86e-4 --ent-coef 0.045 \
    #       --log-std-init -3.66 --gamma 0.9886 --tau 0.00318
    if args.ent_coef      is not None: cfg.ent_coef      = args.ent_coef
    if args.log_std_init  is not None: cfg.log_std_init  = args.log_std_init
    if args.gamma         is not None: cfg.gamma         = args.gamma
    if args.tau           is not None: cfg.tau           = args.tau
    if args.lr            is not None: cfg.learning_rate = args.lr

    # ---- Eval mode: load a checkpoint, run the deterministic learned policy, exit ----
    if args.eval:
        device = args.device if args.device != "auto" else Config.get_device()
        print(f"\n[Eval] Loading residual checkpoint: {args.eval}")
        model = SAC.load(args.eval, device=device,
                         custom_objects={"policy_class": LayerNormSACPolicy})
        avg_dist, best = evaluate_learned(model, args, n_episodes=3, step_label="")
        print(f"\n[Eval] LEARNED policy: avg_dist={avg_dist:.0f}m  "
              f"best_lap={'%.2f' % best if best < float('inf') else 'no lap'}s  "
              f"(DAgger=106.96s)")
        return

    total_timesteps = args.timesteps  or cfg.total_timesteps
    seed_steps      = args.seed_demos or cfg.seed_steps

    print(Config.summary())
    print(f"\n  === Residual RL mode ===")
    print(f"  DAgger base: {args.dagger_weights or cfg.dagger_weights}")
    print(f"  delta:       steer={cfg.delta_steer} (±{cfg.delta_steer*100:.0f}%), accel={cfg.delta_accel} (±{cfg.delta_accel*100:.0f}%)")
    print(f"  Timesteps:   {total_timesteps:,}")

    # ---- Resolve resume ----
    resume_path  = None
    vecnorm_path = None
    reset_timesteps = True

    if args.resume.lower() not in ("", "none", "false"):
        if args.resume == "auto":
            resume_path, buf_path, vecnorm_path = auto_find_resume()
            if resume_path:
                step = _step_count_from_path(resume_path)
                print(f"\n[Resume] Found checkpoint at {step:,} steps")
                reset_timesteps = False
            else:
                print(f"\n[*] No residual checkpoints found — starting fresh.")
        else:
            resume_path = args.resume
            reset_timesteps = False

    # ---- Environment ----
    print(f"\n[Init] Creating ResidualTorcsEnv...")
    if vecnorm_path:
        raw = DummyVecEnv([lambda: ResidualTorcsEnv(
            dagger_weights=args.dagger_weights or cfg.dagger_weights,
        )])
        env = VecNormalize.load(vecnorm_path, raw)
        env.training   = True
        env.norm_reward = False
    else:
        env = make_env(dagger_weights=args.dagger_weights)

    # ---- Model ----
    if resume_path:
        device = args.device if args.device != "auto" else Config.get_device()
        print(f"[Resume] Loading model: {resume_path}")
        model = SAC.load(
            resume_path, env=env, device=device,
            tensorboard_log=Config.LOG_DIR,
            custom_objects={"policy_class": LayerNormSACPolicy},
        )
        if hasattr(args, 'resume_buffer') and args.resume_buffer:
            model.load_replay_buffer(args.resume_buffer)
    else:
        model = create_model(env, args)

    # ---- Pre-training floor check ----
    if not args.skip_floor_check and not resume_path:
        print(f"\n[FloorCheck] Verifying DAgger baseline before any training...")
        floor_ok, dist, lap = verify_floor(model, args, step_label="@init")
        if not floor_ok:
            print(f"\n[ERROR] DAgger baseline is broken before training even starts!")
            print(f"        Re-run eval_policy.py --bc-pretrain {args.dagger_weights or cfg.dagger_weights}")
            print(f"        to diagnose the base policy before training residuals on top.")
            env.close()
            return

    # ---- Seed replay buffer ----
    if seed_steps > 0 and not resume_path:
        seed_replay_buffer(model, env, n_steps=seed_steps, verbose=args.verbose)

    # ---- Telemetry ----
    session_name = args.session_name or f"residual_{time.strftime('%Y%m%d_%H%M%S')}"
    recorder = TelemetryRecorder(session_name=session_name, enabled=not args.no_telemetry)

    callbacks = build_callbacks(args, recorder)

    # ---- Train (single continuous learn — NO chunking) ----
    # Chunking previously crashed because TelemetryCallback closes the CSV at the end
    # of each learn() call. A single learn() closes telemetry exactly once (in finally).
    # The floor is structurally guaranteed (residual=0 is always DAgger), so we don't
    # need periodic floor checks; watch race/best_lap_time in TensorBoard for progress,
    # and use `--eval <checkpoint>` for a deterministic learned-policy evaluation.
    device = args.device if args.device != "auto" else Config.get_device()
    print(f"\n{'='*60}")
    print(f"  Starting Residual SAC Training")
    print(f"  Timesteps: {total_timesteps:,}  |  Device: {device}")
    print(f"  Watch TensorBoard: race/best_lap_time should drop below 106.96s")
    print(f"  Eval a checkpoint anytime: train_residual_sac.py --eval checkpoints/residual_sac_<N>_steps.zip")
    print(f"{'='*60}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            tb_log_name="residual_sac",
            reset_num_timesteps=reset_timesteps,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")

    finally:
        # Always save final checkpoint
        final_path = os.path.join(Config.CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_final")
        model.save(final_path)
        print(f"\n[Save] Model → {final_path}.zip")

        buf_path = os.path.join(Config.CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_final_replay_buffer")
        model.save_replay_buffer(buf_path)

        vn_path = os.path.join(Config.CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_final_vecnorm.pkl")
        env.save(vn_path)

        recorder.close()
        env.close()
        print("[Done] Residual RL training complete.")
        print(f"[Done] Evaluate the learned policy: "
              f"python train_residual_sac.py --eval {final_path}.zip")


if __name__ == "__main__":
    main()

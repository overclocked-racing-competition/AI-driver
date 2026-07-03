# From-scratch SAC with progressive speed-cap curriculum (80/120/160/no-cap km/h),
# teacher-seeded replay, BCAnchoredSAC, gSDE exploration, entropy decay,
# best-deterministic-checkpoint capture. Legacy path (superseded by distillation).
# Usage: python train_sac_v2.py --device cuda | --resume auto | --eval <ckpt.zip>

from __future__ import annotations

import os
import sys
import glob
import time
import argparse
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

from config import Config, TrainingConfig, CHECKPOINT_DIR
from core.torcs_env_sac import TorcsSACEnv
from core.custom_policy import LayerNormSACPolicy
from core.telemetry_recorder import TelemetryRecorder
from core.callbacks import (
    TelemetryCallback, LapTimeCallback, TorcsRelaunchCallback,
    EnhancedCheckpointCallback, FreezeActorCallback,
)

CHECKPOINT_PREFIX = "sac_v2"


# ─────────────────────────────────────────────────────────────────────
#  Speed-cap environment wrapper
# ─────────────────────────────────────────────────────────────────────

class SpeedCapTorcsEnv(TorcsSACEnv):
    # TorcsSACEnv with a progressive speed cap.
    # The cap starts at speed_cap_kmh and is raised after every N clean laps.
    # This implements the curriculum that the 1:36 team used.

    def __init__(self, stage: int = 1, port: int = None,
                 initial_cap_kmh: float = None,
                 increment_kmh:   float = None,
                 laps_per_level:  int   = None,
                 max_cap_kmh:     float = None):
        super().__init__(stage=stage, port=port)
        cfg = Config.training
        self._cap_kmh       = initial_cap_kmh or cfg.speed_cap_start_kmh
        self._increment_kmh = increment_kmh   or cfg.speed_cap_increment_kmh
        self._laps_needed   = laps_per_level  or cfg.speed_cap_episodes_needed
        self._max_cap       = max_cap_kmh     or cfg.speed_cap_max_kmh
        self._laps_at_cap   = 0
        self._cap_active    = self._cap_kmh < self._max_cap
        print(f"[SpeedCap] Initial cap: {self._cap_kmh:.0f} km/h | "
              f"increment: {self._increment_kmh:.0f} km/h | "
              f"laps/level: {self._laps_needed}")

    @property
    def current_cap_kmh(self) -> float:
        return self._cap_kmh

    def _apply_speed_cap(self, action: np.ndarray, speed_kmh: float) -> np.ndarray:
        # Scale throttle down if speed exceeds cap.
        if not self._cap_active or speed_kmh <= self._cap_kmh:
            return action

        steer, accel_brake = action[0], action[1]
        overshoot = (speed_kmh - self._cap_kmh) / max(self._cap_kmh, 10.0)

        if accel_brake > 0:
            # Throttle: reduce proportionally
            new_accel = accel_brake * max(0.0, 1.0 - overshoot * 3.0)
            if overshoot > 0.15:
                new_accel = -0.2 * overshoot   # mild braking when well over cap
            accel_brake = new_accel

        return np.array([steer, accel_brake], dtype=np.float32)

    def step(self, action: np.ndarray):
        # Get raw obs to extract speed
        raw = self.get_raw_obs() if hasattr(self, "get_raw_obs") else {}
        speed = float(raw.get("speedX", 0.0)) if raw else 0.0

        capped_action = self._apply_speed_cap(action, speed)
        obs, reward, terminated, truncated, info = super().step(capped_action)

        # Check for lap completion and promote cap
        if info.get("lap_completed", False):
            self._laps_at_cap += 1
            if self._laps_at_cap >= self._laps_needed:
                old_cap = self._cap_kmh
                self._cap_kmh = min(self._cap_kmh + self._increment_kmh, self._max_cap)
                self._cap_active = self._cap_kmh < self._max_cap
                self._laps_at_cap = 0
                print(f"\n[SpeedCap] ★ Promoted! {old_cap:.0f} → {self._cap_kmh:.0f} km/h "
                      f"({'UNCAPPED' if not self._cap_active else 'capped'})")

        info["speed_cap_kmh"] = self._cap_kmh
        return obs, reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────
#  Environment factory
# ─────────────────────────────────────────────────────────────────────

def make_env(initial_cap_kmh: float = None) -> VecNormalize:
    raw = DummyVecEnv([lambda: SpeedCapTorcsEnv(stage=1,
                                                 initial_cap_kmh=initial_cap_kmh)])
    return VecNormalize(raw, norm_obs=False, norm_reward=False,
                        clip_reward=10.0, gamma=Config.sac.gamma)


# ─────────────────────────────────────────────────────────────────────
#  Model creation
# ─────────────────────────────────────────────────────────────────────

def create_model(env: VecNormalize, args) -> SAC:
    cfg    = Config.sac
    device = args.device if args.device != "auto" else Config.get_device()
    lr     = getattr(args, "lr", None) or cfg.learning_rate

    print(f"\n{'='*60}")
    print(f"  Creating SAC V2 Model (from-scratch, speed-cap curriculum)")
    print(f"  Policy:  LayerNormSACPolicy | Device: {device}")
    print(f"  Arch:    pi={cfg.pi_layers}  qf={cfg.qf_layers}")
    print(f"  LR:      {lr}  |  gSDE: ON  |  sde_sample_freq={cfg.sde_sample_freq}")
    print(f"{'='*60}\n")

    model = SAC(
        policy=LayerNormSACPolicy,
        env=env,
        learning_rate=lr,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        train_freq=(cfg.train_freq, "step"),
        gradient_steps=cfg.gradient_steps,
        ent_coef=cfg.ent_coef,
        target_entropy=cfg.target_entropy,
        learning_starts=cfg.learning_starts,
        use_sde=cfg.use_sde,
        sde_sample_freq=cfg.sde_sample_freq,
        policy_kwargs=dict(
            net_arch=dict(pi=cfg.pi_layers, qf=cfg.qf_layers),
            activation_fn=torch.nn.ReLU,
            use_sde=cfg.use_sde,
        ),
        tensorboard_log=Config.LOG_DIR,
        verbose=getattr(args, "verbose", 1),
        device=device,
    )
    return model


# ─────────────────────────────────────────────────────────────────────
#  Replay buffer seeding from v3 teacher
# ─────────────────────────────────────────────────────────────────────

def seed_buffer_from_teacher(model: SAC, env: VecNormalize,
                              teacher_params_path: str,
                              n_steps: int = 50_000,
                              verbose: int = 1) -> np.ndarray:
    # Seed the replay buffer with v3 teacher rollouts.
    # Returns (observations, actions) for BCAnchoredSAC demo data.
    from agents.teacher_controller_v3 import TeacherController, TeacherV3Params, load_params
    from core.observation_utils import get_observation_dim

    if teacher_params_path and os.path.exists(teacher_params_path):
        teacher_params = load_params(teacher_params_path)
        print(f"[Seed] Using v3 teacher params from: {teacher_params_path}")
    else:
        teacher_params = TeacherV3Params()
        print("[Seed] Using default TeacherV3Params (not tuned).")

    teacher = TeacherController(teacher_params)
    teacher.reset()

    obs_dim   = get_observation_dim(stage=1)
    all_obs   = np.zeros((n_steps, obs_dim), dtype=np.float32)
    all_acts  = np.zeros((n_steps, 2), dtype=np.float32)

    laps = 0
    max_dist = 0.0
    t0 = time.time()

    obs = env.reset()
    raw_obs = env.get_attr("get_raw_obs")[0]() if hasattr(env.envs[0], "get_raw_obs") else {}
    teacher.reset()

    print(f"[Seed] Collecting {n_steps:,} teacher steps for buffer seeding...")

    for step in range(n_steps):
        teacher_action = teacher.act(raw_obs)
        all_obs[step]  = obs[0]
        all_acts[step] = teacher_action

        obs_action = teacher_action.reshape(1, -1)
        next_obs, rewards, dones, infos = env.step(obs_action)

        # Store in replay buffer
        terminal_obs = infos[0].get("terminal_observation", next_obs[0]) if dones[0] else next_obs[0]
        model.replay_buffer.add(
            obs=obs,
            next_obs=np.array([terminal_obs]) if dones[0] else next_obs,
            action=obs_action,
            reward=rewards,
            done=dones,
            infos=infos,
        )

        dist = float(infos[0].get("distRaced", 0.0)) if infos else 0.0
        max_dist = max(max_dist, dist)
        if infos and infos[0].get("lap_completed", False):
            laps += 1
            llt = float(infos[0].get("lastLapTime", 0.0))
            if verbose:
                print(f"[Seed] LAP {laps}: {llt:.2f}s")

        if dones[0]:
            obs = env.reset()
            teacher.reset()
            try:
                raw_obs = env.get_attr("get_raw_obs")[0]()
            except Exception:
                raw_obs = {}
        else:
            obs = next_obs
            try:
                raw_obs = env.get_attr("get_raw_obs")[0]()
            except Exception:
                raw_obs = {}

        if verbose and (step + 1) % 10_000 == 0:
            fps = (step + 1) / max(1.0, time.time() - t0)
            print(f"[Seed] {step+1:,}/{n_steps:,} | laps={laps} | "
                  f"max_dist={max_dist:.0f}m | fps={fps:.0f}")

    elapsed = time.time() - t0
    print(f"[Seed] Done: {laps} laps | {n_steps:,} steps in {elapsed:.0f}s")
    if laps == 0:
        print("[Seed] WARNING: no laps completed. Teacher may need tuning first.")

    model.learning_starts = 0
    return all_obs, all_acts


# ─────────────────────────────────────────────────────────────────────
#  Best-checkpoint callback
# ─────────────────────────────────────────────────────────────────────

class BestLapCheckpointCallback(BaseCallback):
    # Evaluates the deterministic policy every eval_freq steps.
    # Saves the fastest-lapping checkpoint as <prefix>_best.zip.

    def __init__(self, eval_freq: int = 50_000, n_episodes: int = 2,
                 save_prefix: str = None, verbose: int = 1):
        super().__init__(verbose)
        self._eval_freq   = eval_freq
        self._n_episodes  = n_episodes
        self._save_prefix = save_prefix or os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_best")
        self._best_lap    = float("inf")
        self._best_dist   = 0.0
        self._last_eval   = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self._eval_freq:
            return True
        self._last_eval = self.num_timesteps

        env = DummyVecEnv([lambda: TorcsSACEnv(stage=1)])
        lap_times, max_dists = [], []

        obs = env.reset()
        for ep in range(self._n_episodes):
            ep_max, last_lap = 0.0, None
            for _ in range(Config.torcs.max_steps_per_episode):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = env.step(action)
                ep_max = max(ep_max, float(infos[0].get("distRaced", 0.0)))
                if infos[0].get("lap_completed", False):
                    llt = float(infos[0].get("lastLapTime", 0.0))
                    if llt > 0 and llt != last_lap:
                        last_lap = llt
                        lap_times.append(llt)
                if dones[0]:
                    obs = env.reset()
                    break
            max_dists.append(ep_max)
        env.close()

        best   = min(lap_times) if lap_times else float("inf")
        avg_d  = float(np.mean(max_dists))

        improved = False
        if lap_times:
            if best < self._best_lap:
                self._best_lap = best
                improved = True
        elif avg_d > self._best_dist and self._best_lap == float("inf"):
            self._best_dist = avg_d
            improved = True

        tag = f"{self._best_lap:.2f}s" if self._best_lap < float("inf") else f"{self._best_dist:.0f}m"
        if improved:
            self.model.save(self._save_prefix)
            print(f"\n[BestCkpt] ★ New best: {tag} → {self._save_prefix}.zip "
                  f"(@{self.num_timesteps:,} steps)")
        elif self.verbose:
            cur = f"{best:.2f}s" if lap_times else f"{avg_d:.0f}m"
            print(f"[BestCkpt] No improvement ({cur} vs held {tag})")
        return True


# ─────────────────────────────────────────────────────────────────────
#  Entropy decay callback
# ─────────────────────────────────────────────────────────────────────

class EntropyDecayCallback(BaseCallback):
    # Linearly decays ent_coef from init to final over [start, end] steps.

    def __init__(self, ent_init: float, ent_final: float,
                 decay_start: int, decay_end: int, verbose: int = 1):
        super().__init__(verbose)
        self.ent_init   = ent_init
        self.ent_final  = ent_final
        self.decay_start = decay_start
        self.decay_end   = decay_end

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t < self.decay_start:
            return True
        frac    = min(1.0, (t - self.decay_start) / max(1, self.decay_end - self.decay_start))
        new_val = self.ent_init + frac * (self.ent_final - self.ent_init)
        self.model.ent_coef_tensor = torch.tensor(
            new_val, device=self.model.device, dtype=torch.float32
        )
        return True


# ─────────────────────────────────────────────────────────────────────
#  Resume logic
# ─────────────────────────────────────────────────────────────────────

def auto_find_resume():
    pattern = os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_*_steps.zip")
    models  = glob.glob(pattern)
    if not models:
        return None, None, None

    def _steps(p):
        for part in os.path.basename(p).replace(".zip", "").split("_"):
            if part.isdigit():
                return int(part)
        return 0

    latest = max(models, key=_steps)
    step   = _steps(latest)
    base   = os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_{step}")
    buf    = base + "_replay_buffer.pkl"
    vn     = base + "_vecnorm.pkl"
    return latest, buf if os.path.exists(buf) else None, vn if os.path.exists(vn) else None


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="From-Scratch SAC V2 with Speed-Cap Curriculum"
    )
    p.add_argument("--timesteps",    type=int,   default=5_000_000)
    p.add_argument("--device",       type=str,   default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--seed-demos",   type=int,   default=50_000,
                   help="Teacher steps to seed replay buffer (0 to skip)")
    p.add_argument("--teacher-params", type=str, default=None,
                   help="Path to v3 teacher JSON (from optuna_teacher_v3.py export-best)")
    p.add_argument("--initial-cap",  type=float, default=None,
                   help="Initial speed cap km/h (default: Config.training.speed_cap_start_kmh)")
    p.add_argument("--eval-freq",    type=int,   default=50_000,
                   help="Steps between best-checkpoint evals")
    p.add_argument("--resume",       type=str,   default="auto")
    p.add_argument("--eval",         type=str,   default=None,
                   help="Evaluate a checkpoint's deterministic policy and exit")
    p.add_argument("--no-telemetry", action="store_true", default=False)
    p.add_argument("--session-name", type=str,   default=None)
    p.add_argument("--verbose",      type=int,   default=1)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    cfg_t   = Config.training
    device  = args.device if args.device != "auto" else Config.get_device()

    print(Config.summary())
    print(f"\n  === From-Scratch SAC V2 + Speed-Cap Curriculum ===")
    print(f"  Initial speed cap: {args.initial_cap or cfg_t.speed_cap_start_kmh:.0f} km/h")
    print(f"  Target:            sub-80s lap")

    # ── Eval mode ──
    if args.eval:
        print(f"\n[Eval] Loading {args.eval}...")
        model   = SAC.load(args.eval, device=device,
                           custom_objects={"policy_class": LayerNormSACPolicy})
        eval_env = DummyVecEnv([lambda: TorcsSACEnv(stage=1)])
        lap_times, max_dists = [], []
        obs = eval_env.reset()
        for ep in range(3):
            ep_max = 0.0
            for _ in range(Config.torcs.max_steps_per_episode):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = eval_env.step(action)
                ep_max = max(ep_max, float(infos[0].get("distRaced", 0.0)))
                if infos[0].get("lap_completed", False):
                    llt = float(infos[0].get("lastLapTime", 0.0))
                    if llt > 0:
                        lap_times.append(llt)
                        print(f"  Ep{ep+1} LAP: {llt:.3f}s")
                if dones[0]:
                    break
            max_dists.append(ep_max)
        eval_env.close()
        print(f"\n[Eval] best_lap={'%.3f' % min(lap_times) if lap_times else 'no lap'}s"
              f"  avg_dist={np.mean(max_dists):.0f}m")
        return

    # ── Resume ──
    resume_path = vecnorm_path = buf_path = None
    reset_timesteps = True
    if args.resume.lower() not in ("", "none", "false"):
        if args.resume == "auto":
            resume_path, buf_path, vecnorm_path = auto_find_resume()
            if resume_path:
                reset_timesteps = False
                print(f"\n[Resume] Found checkpoint: {resume_path}")
            else:
                print("\n[*] No checkpoint found — fresh start.")
        else:
            resume_path = args.resume
            reset_timesteps = False

    # ── Environment ──
    if vecnorm_path:
        raw = DummyVecEnv([lambda: SpeedCapTorcsEnv(
            stage=1, initial_cap_kmh=args.initial_cap)])
        env = VecNormalize.load(vecnorm_path, raw)
        env.training    = True
        env.norm_reward = False
    else:
        env = make_env(initial_cap_kmh=args.initial_cap)

    # ── Model ──
    if resume_path:
        model = SAC.load(resume_path, env=env, device=device,
                         tensorboard_log=Config.LOG_DIR,
                         custom_objects={"policy_class": LayerNormSACPolicy})
        if buf_path:
            model.load_replay_buffer(buf_path)
            print(f"[Resume] Replay buffer: {model.replay_buffer.size():,} transitions")
    else:
        model = create_model(env, args)

    # ── Seed buffer + demo data ──
    demo_obs = demo_acts = None
    if args.seed_demos > 0 and not resume_path:
        demo_obs, demo_acts = seed_buffer_from_teacher(
            model, env,
            teacher_params_path=args.teacher_params,
            n_steps=args.seed_demos,
            verbose=args.verbose,
        )

    # ── Callbacks ──
    session = args.session_name or f"sac_v2_{time.strftime('%Y%m%d_%H%M%S')}"
    recorder = TelemetryRecorder(session_name=session, enabled=not args.no_telemetry)

    callbacks = [
        LapTimeCallback(verbose=args.verbose),
        TorcsRelaunchCallback(verbose=args.verbose),
        EnhancedCheckpointCallback(
            save_freq=cfg_t.checkpoint_freq,
            save_path=CHECKPOINT_DIR,
            name_prefix=CHECKPOINT_PREFIX,
            save_replay_buffer=True,
            save_vecnorm=True,
            verbose=args.verbose,
        ),
        BestLapCheckpointCallback(
            eval_freq=args.eval_freq,
            n_episodes=2,
            verbose=args.verbose,
        ),
        EntropyDecayCallback(
            ent_init=0.1,
            ent_final=0.01,
            decay_start=500_000,
            decay_end=2_000_000,
            verbose=args.verbose,
        ),
    ]
    if not args.no_telemetry:
        callbacks.insert(0, TelemetryCallback(recorder=recorder,
                                               verbose=args.verbose))

    print(f"\n{'='*60}")
    print(f"  Training SAC V2 | {args.timesteps:,} steps | {device}")
    print(f"  Speed cap starts at {args.initial_cap or cfg_t.speed_cap_start_kmh:.0f} km/h")
    print(f"  Best checkpoint: {CHECKPOINT_PREFIX}_best.zip")
    print(f"{'='*60}\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(callbacks),
            log_interval=10,
            tb_log_name="sac_v2",
            reset_num_timesteps=reset_timesteps,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[!] Training interrupted.")
    finally:
        final = os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_final")
        model.save(final)
        model.save_replay_buffer(final + "_replay_buffer")
        env.save(final + "_vecnorm.pkl")
        recorder.close()
        env.close()
        print(f"\n[Done] Model saved → {final}.zip")
        print(f"[Done] Evaluate: python train_sac_v2.py --eval {final}.zip")


if __name__ == "__main__":
    main()

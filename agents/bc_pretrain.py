# Behavioral Cloning Pretraining for SAC Racing AI
# Collects expert (obs, action) pairs from TeacherController,
# trains a feedforward network via MSE loss.

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

from config import Config, PROJECT_ROOT
from core.torcs_env_sac import TorcsSACEnv
from core.observation_utils import get_observation_dim
# tune_teacher is only used for v1/v2 — v3 uses its own JSON format
try:
    import agents.tune_teacher as tune_teacher
    from agents.tune_teacher import load_params_from_json
except ImportError:
    tune_teacher = None
    load_params_from_json = None

# obs[31] = prev_steer; noise here breaks the copycat shortcut
_PREV_STEER_IDX = 31


def build_teacher(controller_version: str, teacher_params_path: str = None):
    # Build the teacher controller (v1/v2/v3/v6)
    if controller_version == "v6":
        import json
        from agents.teacher_controller_v6 import TeacherController, TeacherParamsV6, params_from_optuna
        if teacher_params_path and os.path.exists(teacher_params_path):
            with open(teacher_params_path) as f:
                d = json.load(f)
            params = params_from_optuna(d)
            print(f"[BC] Loaded v6 teacher params from: {teacher_params_path}")
        else:
            params = TeacherParamsV6()
            print("[BC] Using default TeacherParamsV6 (not Optuna-tuned).")
        return TeacherController(params)

    if controller_version == "v3":
        from agents.teacher_controller_v3 import TeacherController, TeacherV3Params, load_params
        if teacher_params_path and os.path.exists(teacher_params_path):
            params = load_params(teacher_params_path)
            print(f"[BC] Loaded v3 teacher params from: {teacher_params_path}")
        else:
            params = TeacherV3Params()
            print(f"[BC] Using default TeacherV3Params (not Optuna-tuned).")
            print("[BC] TIP: run optuna_teacher_v3.py first for best results.")
        return TeacherController(params)

    # v1/v2 legacy path
    if tune_teacher is None:
        raise ImportError("tune_teacher.py not found — cannot build v1/v2 teacher.")
    tune_teacher._set_controller(controller_version)

    if controller_version == "v2":
        import agents.teacher_controller_v2 as tc
    else:
        import agents.teacher_controller as tc

    if teacher_params_path and os.path.exists(teacher_params_path):
        params = load_params_from_json(teacher_params_path)
        print(f"[BC] Loaded {controller_version} teacher params from: {teacher_params_path}")
    else:
        params = tc.TeacherParams()
        print(f"[BC] Using default {controller_version} TeacherParams (not Optuna-tuned).")
        print("[BC] TIP: run tune_teacher.py + --mode export first for best results.")
    return tc.TeacherController(params)


# BC Network — matches SAC actor architecture exactly
class BCNetwork(nn.Module):
    # LayerNorm BC network mirroring LayerNormSACPolicy's actor

    def __init__(self, obs_dim: int, action_dim: int = 2, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# Data Collection
def collect_teacher_data(
    n_steps: int,
    teacher,
    stage: int = 1,
    verbose: bool = True,
) -> tuple:
    # Run teacher in TorcsSACEnv, record (obs, action) pairs.
    # Syncs Config.aids to teacher params to avoid gear/TCS mismatch.
    from config import Config
    _aids_saved = {
        "tcs_enabled":    Config.aids.tcs_enabled,
        "rpm_upshift":    Config.aids.rpm_upshift,
        "rpm_downshift":  Config.aids.rpm_downshift,
    }
    # Disable env-side TCS (teacher applies its own)
    Config.aids.tcs_enabled = False
    # Match gear thresholds to teacher's Optuna-tuned values
    if hasattr(teacher, "p"):  # v5/v6 store params in self.p
        Config.aids.rpm_upshift   = float(teacher.p.rpm_upshift)
        Config.aids.rpm_downshift = float(teacher.p.rpm_downshift)

    env = TorcsSACEnv(stage=stage)
    obs_dim = get_observation_dim(stage)

    observations = np.zeros((n_steps, obs_dim), dtype=np.float32)
    actions      = np.zeros((n_steps, 2),       dtype=np.float32)

    laps      = 0
    max_dist  = 0.0
    t0        = time.time()

    obs_vec, info = env.reset()
    raw_obs = info.get("raw_obs", {})
    teacher.reset()

    for step in range(n_steps):
        action = teacher.act(raw_obs)
        observations[step] = obs_vec
        actions[step]      = action

        obs_vec, _, terminated, truncated, info = env.step(action)
        raw_obs = info.get("raw_obs", {})

        dist = float(raw_obs.get("distRaced", 0.0))
        max_dist = max(max_dist, dist)
        if info.get("lap_completed", False):
            laps += 1
            llt = float(info.get("lastLapTime", 0.0))
            if verbose:
                print(f"  [Collect] LAP {laps}: {llt:.2f}s at step {step:,}")

        if terminated or truncated:
            obs_vec, info = env.reset()
            raw_obs = info.get("raw_obs", {})
            teacher.reset()

        if verbose and (step + 1) % 10000 == 0:
            elapsed = time.time() - t0
            fps = (step + 1) / max(1.0, elapsed)
            print(f"  [Collect] {step+1:7,}/{n_steps:,} | "
                  f"fps: {fps:.0f} | max_dist: {max_dist:.0f}m | laps: {laps}")

    env.close()
    # Restore Config.aids
    Config.aids.tcs_enabled   = _aids_saved["tcs_enabled"]
    Config.aids.rpm_upshift   = _aids_saved["rpm_upshift"]
    Config.aids.rpm_downshift = _aids_saved["rpm_downshift"]
    elapsed = time.time() - t0

    return observations, actions, {
        "n_steps": n_steps,
        "laps": laps,
        "max_dist": max_dist,
        "elapsed_s": elapsed,
        "fps": n_steps / max(1.0, elapsed),
    }


# Training
def _weighted_mse(pred, target, steer_weight_k: float):
    # Per-sample MSE weighted by steering magnitude (emphasize corners)
    se = (pred - target).pow(2).mean(dim=1)
    w  = 1.0 + steer_weight_k * target[:, 0].abs()
    return (se * w).mean()


def train_bc(
    observations: np.ndarray,
    actions: np.ndarray,
    obs_dim: int,
    action_dim: int = 2,
    n_epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 10,
    steer_weight_k: float = 4.0,
    prev_steer_noise: float = 0.15,
    device: str = "auto",
    verbose: bool = True,
) -> BCNetwork:
    # Train BCNetwork via supervised regression on expert data
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    N = len(observations)
    n_val   = int(N * val_split)
    n_train = N - n_val

    perm      = np.random.permutation(N)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    obs_train = torch.tensor(observations[train_idx], dtype=torch.float32).to(device)
    act_train = torch.tensor(actions[train_idx],      dtype=torch.float32).to(device)
    obs_val   = torch.tensor(observations[val_idx],   dtype=torch.float32).to(device)
    act_val   = torch.tensor(actions[val_idx],        dtype=torch.float32).to(device)

    model     = BCNetwork(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    if verbose:
        print(f"\n[BC] Training on {device} | obs_dim={obs_dim}")
        print(f"[BC] {n_train:,} train / {n_val:,} val  |  epochs={n_epochs}  batch={batch_size}")
        print(f"[BC] Corner-weighted loss: steer_weight_k={steer_weight_k}")
        print(f"[BC] prev_steer noise: std={prev_steer_noise}")
        print()

    for epoch in range(n_epochs):
        model.train()
        perm_e     = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_train, batch_size):
            idx        = perm_e[start:start + batch_size]
            batch_obs  = obs_train[idx]
            if prev_steer_noise > 0.0 and batch_obs.shape[1] > _PREV_STEER_IDX:
                batch_obs = batch_obs.clone()
                noise = torch.randn(batch_obs.shape[0], device=batch_obs.device) * prev_steer_noise
                batch_obs[:, _PREV_STEER_IDX] = torch.clamp(
                    batch_obs[:, _PREV_STEER_IDX] + noise, -1.0, 1.0)
            pred     = model(batch_obs)
            loss     = _weighted_mse(pred, act_train[idx], steer_weight_k)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        model.eval()
        with torch.no_grad():
            val_loss = _weighted_mse(model(obs_val), act_val, steer_weight_k).item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | "
                  f"train {epoch_loss/max(n_batches,1):.6f} | val {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"\n[BC] Best val_loss = {best_val_loss:.6f}")

    return model


# Main
def main():
    parser = argparse.ArgumentParser(description="BC Pretraining for SAC Racing AI")
    parser.add_argument("--n-steps", type=int, default=500_000,
                        help="Steps to collect from teacher (default 500k for v3)")
    parser.add_argument("--output", type=str, default="checkpoints/bc_pretrained.pth")
    parser.add_argument("--teacher-params", type=str, default=None,
                        help="Path to JSON file from tune_teacher.py --mode export. "
                             "If omitted, uses default TeacherParams (untuned).")
    parser.add_argument("--controller", type=str, default="v6",
                        choices=["v1", "v2", "v3", "v6"],
                        help="Which teacher controller to imitate (default: v6, the fastest).")
    parser.add_argument("--stage", type=int, default=1, choices=[1])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--prev-steer-noise", type=float, default=0.15,
                        help="Std of noise on prev_steer obs during training "
                             "(breaks the BC copycat shortcut; 0 disables).")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    obs_dim     = get_observation_dim(args.stage)
    output_path = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    teacher = build_teacher(args.controller, args.teacher_params)

    print("\n" + "=" * 55)
    print("  Behavioral Cloning Pretraining")
    print(f"  Steps:      {args.n_steps:,}")
    print(f"  Controller: {args.controller}")
    print(f"  obs_dim:    {obs_dim} (includes prev_steer at dim 31)")
    print("=" * 55)

    observations, actions, stats = collect_teacher_data(
        n_steps=args.n_steps,
        teacher=teacher,
        stage=args.stage,
        verbose=True,
    )

    print(f"\n  Collection: {stats['laps']} laps | "
          f"max_dist={stats['max_dist']:.0f}m | {stats['fps']:.0f} fps")

    if stats["laps"] == 0:
        print("  WARNING: teacher completed no laps.")

    model = train_bc(
        observations=observations,
        actions=actions,
        obs_dim=obs_dim,
        action_dim=2,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        prev_steer_noise=args.prev_steer_noise,
        device=args.device,
        verbose=True,
    )

    torch.save(model.state_dict(), output_path)
    print(f"\n[BC] Saved -> {output_path}")


if __name__ == "__main__":
    main()

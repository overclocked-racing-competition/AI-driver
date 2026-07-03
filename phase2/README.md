# Phase 2 — Mass Start (multi-car racing)

Opponent-aware residual on top of the frozen Phase-1 network. The fast time-trial
policy (`bc_v6.pth`, 32-dim) transfers **losslessly** as a frozen base because the
32-dim observation is a strict prefix of the 68-dim stage-2 observation (32 self +
36 opponent rangefinders). A SAC residual reads the full 68-dim vector and learns
corrections for avoidance, overtaking, and defense:

```
final_action = clip( base(obs[:32]) + delta * residual(obs68), -1, 1 )
```

At `residual = 0` the agent drives the exact Phase-1 racing line — collapse is
structurally impossible, as in Phase 1.

## Files

| File | Role |
|---|---|
| `race_config.py` | generate/install a multi-car Corkscrew race (our `scr_server` + N bots) |
| `residual_env_stage2.py` | `TorcsSACEnvStage2` (68-dim, grid launch) + `Stage2ResidualEnv` (frozen 32-dim base + residual) |
| `train_mass_start.py` | residual SAC trainer + `OpponentRampCallback` (opponent-count curriculum) |
| `submit_agent_stage2.py` | competition submission (base + residual over UDP) |
| `race_configs/` | generated race XML(s) |

Reuses `create_model` / `seed_replay_buffer` from the root `train_residual_sac.py`
(env-agnostic), so the gentle gSDE init and residual=0 seeding match Phase 1.

## Run (headless WSL)

```bash
cd /mnt/d/IBM_competition/SAC/S3_B/S4-F
source /home/user/torcs-venv/bin/activate

# 0) inspect a generated grid (optional)
python3 phase2/race_config.py --opponents 4 --bot inferno

# 1) train the opponent-aware residual on the frozen fast base
HOME=/home/user/torcs_w0 python3 phase2/train_mass_start.py \
    --base-weights checkpoints/bc_v6.pth \
    --start-opponents 2 --max-opponents 8 --ramp-every 150000 \
    --timesteps 1000000

# 2) submit (multi-car TORCS race running on port 3001)
python phase2/submit_agent_stage2.py \
    --base checkpoints/bc_v6.pth --residual checkpoints/massstart_sac_final.zip
```

## Curriculum

`OpponentRampCallback` raises the opponent count `--start-opponents -> --max-opponents`
by one every `--ramp-every` steps. The new grid takes effect on the next TORCS
relaunch (every `Config.torcs.relaunch_every_n_episodes`). Start easy (few cars),
finish dense.

## Requires live TORCS testing (built, not yet run)

This pipeline is code-complete and compile-checked but has **not** been run against
a live multi-car TORCS. Verify on first launch:

1. **Bots load and drive.** `inferno` must exist and be drivable on `corkscrew`;
   if not, try `--bot berniw` / `bt` / `olethros`. Bots race their default cars
   (mixed grid) — fine for training opponent avoidance; the opponent rangefinders
   are car-agnostic.
2. **Grid spawns without overlap.** If cars collide at spawn, reduce opponents or
   widen the grid (`rows` / spacing in `race_config.build_massstart_xml`).
3. **`opponents` telemetry is populated.** `snakeoil3_gym` must parse the SCR
   `opponents` field (36 values); otherwise stage-2 obs degenerates to stage-1
   (all opponents read 200 m). Confirm non-200 values when a bot is near.
4. **Grid start.** Our car starts mid-pack; watch the first ~100 m for contact.
   `driving_aids` launch-centering handles standing-start steering.

## Deployment note

For a clean submission you may distill the trained `base + residual` back into a
single 68-dim `BCNetwork` (as in Phase 1) so the submission stays a plain network
without SB3 at inference. Otherwise `submit_agent_stage2.py` loads both directly.

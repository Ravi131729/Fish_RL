# Fish_RL

Reinforcement learning for fish-robot path tracking using a custom-built dynamics model and simulator, where the policy outputs PID gains (heading + speed) and the environment applies those gains through classical controllers.

## What this repo does

- Simulates fish dynamics and path-following behavior in JAX.
- Trains a PPO policy (`fish/training/train_ppo.py`) that outputs 6 normalized PID gain actions:
  - `kp`, `kd`, `ki` (heading)
  - `v_kp`, `v_kd`, `v_ki` (forward-speed)
- Logs training/evaluation metrics and plots to Weights & Biases.
- Saves checkpoints to `./checkpoints/final`.

## Repository layout

```text
fish/
  agents/        # PPO agent + policy/value networks
  env/           # dynamics, reset, observations, rewards, action parsing
  training/      # rollout, eval rollout, training loop
  configs/       # YAML configs (main: ppo_fish.yaml)
  utils/         # config loading, normalization, checkpoint helpers
scripts/
  train_ppo.py   # older/experimental training entrypoint
  test_mujoco.py # PPO sanity checks on Gymnasium MuJoCo envs
path_vis.py      # random path generation visualization
notes.md         # deployment and controller notes
```

## Setup

1. Create and activate a Python environment.
2. Install dependencies (adjust JAX install for your CPU/GPU):

```bash
pip install --upgrade pip
pip install jax jaxlib flax optax orbax-checkpoint chex numpy matplotlib pyyaml wandb
```

If you use GPU/TPU, install JAX from the official wheel matching your platform.

## Training

Main training entrypoint:

```bash
python -m fish.training.train_ppo
```

Configuration file:

- `fish/configs/ppo_fish.yaml`

Key defaults in config:

- `training.n_envs: 512`
- `training.horizon: 256`
- `training.updates: 2000`
- `env.dim: 6` (action dimension)

## Outputs

- Checkpoints: `checkpoints/final/`
- W&B logs/plots: configured by `logging.project` in `fish/configs/ppo_fish.yaml`

## Experimental results

Add your result GIFs here later (example paths shown):

```markdown
### Experiment 1: Path tracking behavior
![Experiment 1](experiments/exp1.gif)

### Experiment 2: Controller response
![Experiment 2](experiments/exp2.gif)
```

## Notes

- Policy actions are normalized in `[-1, 1]` and mapped to physical PID gains in `fish/env/action_parser.py`.
- Observation/reward/controller details are documented in `notes.md`.

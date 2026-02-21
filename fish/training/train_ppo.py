# ===== core =====
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from flax import nnx

from fish.env.reset import reset_env
from fish.env.observation import build_obs
from fish.agents.ppo_agent import (
    PPO,
    compute_gae,
    ppo_update_with_minibatches,
)

from fish.training.rollout import make_jitted_rollout

from fish.utils.obs_normalizer import RunningMeanStd
from fish.utils.config_loader import load_config


def init_system(seed):

    env_cfg, eval_cfg, ppo_cfg, train_cfg, log_cfg, action_cfg, seed = load_config(
        "fish/configs/ppo_fish.yaml"
    )

    cfg = env_cfg

    N = train_cfg["n_envs"]
    N_eval = train_cfg["n_eval_envs"]
    T = train_cfg["horizon"]
    num_updates = train_cfg["updates"]
    nx = train_cfg["nx"]

    act_dim = action_cfg["dim"]

    wandb.init(
        project=log_cfg["project"],
        config={
            "env": train_cfg,
            "ppo": ppo_cfg.__dict__,
            "action_dim": act_dim,
        },
    )

    key = jax.random.PRNGKey(seed)
    key, key_env, key_agent, key_eval = jax.random.split(key, 4)

    state = reset_env(key_env, N, nx, cfg=cfg)
    state_eval = reset_env(key_env, N_eval, nx, cfg=eval_cfg)

    obs0 = build_obs(state, cfg, key)
    obs_dim = obs0.shape[1]


    agent = PPO(obs_dim, act_dim, rng=key_agent)

    obs_normalizer = RunningMeanStd(obs_dim)

    graphdef, state_vars = nnx.split(agent)
    rollout_fn = make_jitted_rollout(graphdef, state_vars, cfg, T, N)

    return (
        cfg, eval_cfg, ppo_cfg,
        state, state_eval,
        agent, obs_normalizer,
        graphdef, state_vars,
        rollout_fn,
        key, key_eval,
        N, nx, T, num_updates
    )

def collect_rollout(rollout_fn, state, key, state_vars, obs_normalizer):

  obs_mean = jnp.array(obs_normalizer.mean, dtype=jnp.float32)
  obs_std = jnp.sqrt(jnp.array(obs_normalizer.var, dtype=jnp.float32) + 1e-8)

  buffer, state, key, state_vars = rollout_fn(
      state, key, state_vars, obs_mean, obs_std
  )

  obs_normalizer.update(np.asarray(buffer.obs))

  return buffer, state, key, state_vars

def update_policy(agent, buffer, state, obs_normalizer, ppo_cfg, graphdef, state_vars, key, cfg):

    last_obs = build_obs(state, cfg, key)
    last_obs_norm = obs_normalizer.normalize(last_obs)
    last_value = agent.critic(last_obs_norm)

    advantages, returns = compute_gae(buffer, last_value, ppo_cfg)

    T_, N_ = advantages.shape

    obs_norm_flat = obs_normalizer.normalize(buffer.obs.reshape(T_*N_, -1))
    act_flat = buffer.actions.reshape(T_*N_, -1)
    logp_flat = buffer.log_probs.reshape(T_*N_)
    val_flat = buffer.values.reshape(T_*N_)
    adv_flat = advantages.reshape(T_*N_)
    ret_flat = returns.reshape(T_*N_)

    batch_data = (obs_norm_flat, act_flat, logp_flat, adv_flat, ret_flat, val_flat)

    key, key_update = jax.random.split(key)
    state_vars, metrics = ppo_update_with_minibatches(
        graphdef,
        state_vars,
        batch_data,
        key_update,
        ppo_cfg,
    )

    return state_vars, metrics, key

def log_metrics(metrics, buffer, update, num_updates):

    ep_reward_mean = float(buffer.rewards.sum(axis=0).mean())

    wandb.log({
        "loss_total": float(metrics['loss_total']),
        "loss_pi": float(metrics['loss_pi']),
        "loss_v": float(metrics['loss_v']),
        "entropy": float(metrics['entropy']),
        "ep_reward": ep_reward_mean,
        "update": update
    })

    if update % 10 == 0:
        print(f"update {update}/{num_updates} reward={ep_reward_mean:.2f}")


def main(seed=10):

    (
        cfg, eval_cfg, ppo_cfg,
        state, state_eval,
        agent, obs_normalizer,
        graphdef, state_vars,
        rollout_fn,
        key, key_eval,
        N, nx, T, num_updates
    ) = init_system(seed)

    for update in range(num_updates):

        buffer, state, key, state_vars = collect_rollout(
            rollout_fn, state, key, state_vars, obs_normalizer
        )

        state_vars, metrics, key = update_policy(
            agent, buffer, state,
            obs_normalizer, ppo_cfg,
            graphdef, state_vars, key, cfg
        )

        log_metrics(metrics, buffer, update, num_updates)

    print("Training done")

if __name__ == "__main__":
    main(seed=10)
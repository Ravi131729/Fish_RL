import jax
import jax.numpy as jnp
from flax import nnx
import chex
import optax
from functools import partial
from typing import Tuple, Optional

from fish.agents.networks import Actor, ValueNet, sample_action, compute_log_prob
# from fish.env.fish_env import (
#     EnvState, EnvConfig,
#     reset_env, step_env,
#     build_obs, compute_reward
# )
from fish.env.observation import build_obs
from fish.env.reward import compute_reward
from fish.env.action_parser import parse_action
from fish.env.reset import reset_env, sample_physics_params
from fish.env.types import EnvState, EnvConfig
from fish.env.env_fish import step_env

@chex.dataclass
class RolloutBuffer:
    """Storage for PPO rollout data using chex dataclass."""
    obs: jnp.ndarray        # (T, N, obs_dim)
    actions: jnp.ndarray    # (T, N, act_dim)
    log_probs: jnp.ndarray  # (T, N)
    values: jnp.ndarray     # (T, N)
    rewards: jnp.ndarray    # (T, N)
    dones: jnp.ndarray      # (T, N)
    avg_forward_velocity: jnp.ndarray  # (T, N)
    avg_lateral_velocity: jnp.ndarray  # (T, N)
    qh: jnp.ndarray      # (T, N)
    avg_heading: jnp.ndarray      # (T, N)
    omega_avg: jnp.ndarray      # (T, N) - added for debugging
    u: jnp.ndarray      # (T, N) - added for debugging
    qdh: jnp.ndarray      # (T, N) - added for debugging
    ux: jnp.ndarray      # (T, N) - added for debugging
    uy: jnp.ndarray      # (T, N) - added for debugging
    # velocity_error: jnp.ndarray      # (T, N) - added for debugging
    heading_error: jnp.ndarray      # (T, N) - added for debugging
    cross_track_error: jnp.ndarray      # (T, N) - added for debugging



@chex.dataclass
class PPOConfig:
    """PPO hyperparameters."""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.001
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    n_minibatches: int = 4
    normalize_advantage: bool = True
    target_kl: float = 0.02  # Target KL for early stopping (tightened)


def compute_gae(
    buffer: RolloutBuffer,
    last_value: jnp.ndarray,
    ppo_cfg: PPOConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GAE advantages and returns."""
    T, N = buffer.rewards.shape

    # Append bootstrap value
    values_extended = jnp.concatenate([buffer.values, last_value[None, :]], axis=0)

    def gae_step(carry, t):
        gae = carry
        delta = (
            buffer.rewards[t]
            + ppo_cfg.gamma * values_extended[t + 1] * (1 - buffer.dones[t])
            - values_extended[t]
        )
        gae = delta + ppo_cfg.gamma * ppo_cfg.gae_lambda * (1 - buffer.dones[t]) * gae
        return gae, gae

    # Scan backwards
    indices = jnp.arange(T - 1, -1, -1)
    _, advantages = jax.lax.scan(
        gae_step,
        jnp.zeros(N),
        indices,
    )
    advantages = advantages[::-1]  # Reverse to get correct order

    returns = advantages + buffer.values
    return advantages, returns
@jax.jit
def ppo_update_step(graphdef, state, mini_batch, rng, ppo_cfg: PPOConfig):
    model = nnx.merge(graphdef, state)
    obs, act, logp_old, adv, ret, v_old = mini_batch

    # Advantage normalization (very common)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    def loss_fn(params, obs, act, logp_old, adv, ret, v_old, rng):
        actor, critic = params

        # Actor produces mean and log_std; compute log-prob of the STORED action
        # (not a new sample) to get the correct importance ratio for PPO.
        mean, log_std = actor(obs)
        logp = compute_log_prob(mean, log_std, act)

        # ratio = exp(logπ_new - logπ_old), clip for numerical stability
        log_ratio = logp - logp_old
        log_ratio = jnp.clip(log_ratio, -20.0, 20.0)  # prevent overflow
        ratio = jnp.exp(log_ratio)
        unclipped = ratio * adv
        clipped = jnp.clip(
            ratio,
            1.0 - ppo_cfg.clip_eps,
            1.0 + ppo_cfg.clip_eps,
        ) * adv
        policy_loss = -(jnp.minimum(unclipped, clipped)).mean()

        # Value loss (with clipping)
        v = critic(obs)

        v_clipped = v_old + jnp.clip(
            v - v_old,
            -ppo_cfg.clip_eps,
            ppo_cfg.clip_eps,
        )
        vf_loss1 = (v - ret) ** 2
        vf_loss2 = (v_clipped - ret) ** 2
        value_loss = 0.5 * jnp.maximum(vf_loss1, vf_loss2).mean()

        # Entropy bonus (approximate, based on Gaussian log_std before tanh)
        entropy = (log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)).sum(axis=-1).mean()
        entropy_loss = -ppo_cfg.ent_coef * entropy

        total = policy_loss + ppo_cfg.vf_coef * value_loss + entropy_loss

        # Better KL approximation: (ratio - 1) - log(ratio)
        # This is more accurate than the quadratic approximation
        log_ratio_clipped = jnp.clip(log_ratio, -20.0, 20.0)
        approx_kl = ((ratio - 1) - log_ratio_clipped).mean()

        metrics = {
            "loss_total": total,
            "loss_pi": policy_loss,
            "loss_v": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clipfrac": (jnp.abs(ratio - 1.0) > ppo_cfg.clip_eps).mean(),
        }
        return total, metrics

    params = (model.actor, model.critic)

    (loss, metrics), grads = nnx.value_and_grad(
        loss_fn,
        has_aux=True,
    )(params, obs, act, logp_old, adv, ret, v_old, rng)

    grads_actor, grads_critic = grads  # grads has same structure as params

    model.actor_opt.update(model.actor, grads_actor)
    model.critic_opt.update(model.critic, grads_critic)

    graphdef, state = nnx.split(model)
    return state, metrics


def ppo_update_with_minibatches(
    graphdef,
    state_vars,
    batch_data: Tuple,
    rng: jax.Array,
    ppo_cfg: PPOConfig,
) -> Tuple:
    """
    Perform multiple epochs of PPO updates with minibatches.

    Args:
        graphdef: nnx graph definition
        state_vars: nnx state variables
        batch_data: Tuple of (obs, act, logp_old, adv, ret, v_old) - all flattened
        rng: random key
        ppo_cfg: PPO configuration

    Returns:
        Updated state_vars and aggregated metrics
    """
    obs, act, logp_old, adv, ret, v_old = batch_data
    batch_size = obs.shape[0]
    minibatch_size = batch_size // ppo_cfg.n_minibatches

    # Aggregate metrics across all updates
    all_metrics = {
        "loss_total": 0.0,
        "loss_pi": 0.0,
        "loss_v": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clipfrac": 0.0,
    }
    n_updates = 0
    early_stop = False

    for epoch in range(ppo_cfg.n_epochs):
        if early_stop:
            break

        # Shuffle data at each epoch
        rng, shuffle_key = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_key, batch_size)
        obs_shuffled = obs[perm]
        act_shuffled = act[perm]
        logp_shuffled = logp_old[perm]
        adv_shuffled = adv[perm]
        ret_shuffled = ret[perm]
        v_old_shuffled = v_old[perm]

        for mb_idx in range(ppo_cfg.n_minibatches):
            if early_stop:
                break

            start = mb_idx * minibatch_size
            end = start + minibatch_size

            mini_batch = (
                obs_shuffled[start:end],
                act_shuffled[start:end],
                logp_shuffled[start:end],
                adv_shuffled[start:end],
                ret_shuffled[start:end],
                v_old_shuffled[start:end],
            )

            rng, update_key = jax.random.split(rng)
            state_vars, metrics = ppo_update_step(
                graphdef, state_vars, mini_batch, update_key, ppo_cfg
            )

            # Accumulate metrics
            for k in all_metrics:
                all_metrics[k] += float(metrics[k])
            n_updates += 1

            # Early stopping based on KL divergence
            if float(metrics["approx_kl"]) > 1.5 * ppo_cfg.target_kl:
                early_stop = True
                break

    # Average metrics
    for k in all_metrics:
        all_metrics[k] /= max(n_updates, 1)

    all_metrics["n_updates"] = n_updates
    all_metrics["early_stopped"] = early_stop

    return state_vars, all_metrics


def gaussian_log_prob(mean, log_std, act):
    # Legacy helper kept for reference; not used in current tanh-policy setup.
    var = jnp.exp(2.0 * log_std)
    logp_per_dim = -0.5 * (((act - mean) ** 2) / var + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    return logp_per_dim.sum(axis=-1)
import optax

class PPO(nnx.Module):
    def __init__(self, obs_dim, act_dim, rng, lr_actor=1e-4, lr_critic=3e-4, max_grad_norm=0.5):
        ka, kv = jax.random.split(rng, 2)
        self.actor = Actor(obs_dim, act_dim, rngs=nnx.Rngs(ka))
        self.critic = ValueNet(obs_dim, rngs=nnx.Rngs(kv))

        # Optimizers with gradient clipping
        actor_opt = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(lr_actor)
        )
        critic_opt = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(lr_critic)
        )
        self.actor_opt  = nnx.Optimizer(self.actor, actor_opt, wrt=nnx.Param)
        self.critic_opt = nnx.Optimizer(self.critic, critic_opt, wrt=nnx.Param)


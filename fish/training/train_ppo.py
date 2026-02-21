# ===== core =====
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from flax import nnx
import os
import orbax.checkpoint as ocp

from fish.env.reset import reset_env
from fish.env.observation import build_obs
from fish.agents.ppo_agent import (
    PPO,
    compute_gae,
    ppo_update_with_minibatches,
)

from fish.training.rollout import make_jitted_rollout
from fish.training.eval_rollout import make_eval_rollout
from fish.utils.obs_normalizer import RunningMeanStd
from fish.utils.config_loader import load_config
import matplotlib.pyplot as plt
import time
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
    eval_rollout_fn = make_eval_rollout(graphdef, eval_cfg, T)

    checkpoint_dir = os.path.abspath("./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpointer = ocp.StandardCheckpointer()

    return (
        cfg, eval_cfg, ppo_cfg,
        state, state_eval,
        agent, obs_normalizer,
        graphdef, state_vars,
        rollout_fn,
        eval_rollout_fn,
        key, key_eval,
        N,N_eval, nx, T, num_updates ,checkpoint_dir, checkpointer
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
def log_metrics(metrics, buffer, update, num_updates, update_time, start_time):

    # ===== reward stats =====
    ep_rewards = buffer.rewards.sum(axis=0)
    ep_reward_mean = float(ep_rewards.mean())
    ep_reward_std  = float(ep_rewards.std())
    reward_per_step = float(buffer.rewards.mean())

    # ===== state stats =====
    avg_forward_velocity_mean = float(buffer.ux.mean())
    avg_lateral_velocity_mean = float(buffer.uy.mean())
    qh_mean = float(buffer.qh.mean())
    avg_heading_mean = float(buffer.avg_heading.mean())
    omega_avg_mean = float(buffer.omega_avg.mean())

    # ===== errors =====
    heading_error_mean = float(buffer.heading_error.mean())
    cross_track_error_mean = float(buffer.cross_track_error.mean())

    elapsed = time.time() - start_time

    wandb.log({

        # ---------- PPO ----------
        "train/loss_total": float(metrics['loss_total']),
        "train/loss_pi": float(metrics['loss_pi']),
        "train/loss_v": float(metrics['loss_v']),
        "train/entropy": float(metrics['entropy']),
        "train/approx_kl": float(metrics.get('approx_kl', 0.0)),
        "train/clipfrac": float(metrics.get('clipfrac', 0.0)),
        "train/n_updates": metrics.get('n_updates', 0),
        "train/early_stopped": int(metrics.get('early_stopped', 0)),

        # ---------- reward ----------
        "reward/ep_reward_mean": ep_reward_mean,
        "reward/ep_reward_std": ep_reward_std,
        "reward/reward_per_step": reward_per_step,

        # ---------- state stats ----------
        "state/avg_forward_velocity": avg_forward_velocity_mean,
        "state/avg_lateral_velocity": avg_lateral_velocity_mean,
        "state/qh_mean": qh_mean,
        "state/avg_heading": avg_heading_mean,
        "state/omega_avg": omega_avg_mean,

        # ---------- errors ----------
        "error/heading_error_mean": heading_error_mean,
        "error/cross_track_error_mean": cross_track_error_mean,

        # ---------- time ----------
        "time/update_time": update_time,
        "time/elapsed": elapsed,

        "update": update + 1,
    })

    if update % 10 == 0:
        print(f"update {update}/{num_updates} reward={ep_reward_mean:.2f}")

    return ep_reward_mean
def run_eval_and_log(
    graphdef,
    state_vars,
    obs_normalizer,
    eval_cfg,
    key_eval,
    N_eval,
    nx,
    eval_every_step,
):


    # --- reset eval env ---
    state_eval = reset_env(key_eval, N_eval, nx, eval_cfg)
    eval_fn = make_eval_rollout(graphdef, eval_cfg, T=eval_cfg.max_steps - 1)

    obs_mean = jnp.array(obs_normalizer.mean, dtype=jnp.float32)
    obs_std = jnp.sqrt(jnp.array(obs_normalizer.var, dtype=jnp.float32) + 1e-8)

    traj, state_eval, key_eval = eval_fn(
        state_eval,
        key_eval,
        state_vars,
        obs_mean,
        obs_std
    )



    # ============================================================
    #  PLOTS
    # ============================================================

    i = 0  # first env only

    # ---- Trajectory ----
    x = traj["x"]
    y = traj["y"]

    traj_fig = plt.figure(figsize=(6,6))
    i = 0

    path_x = traj["path_x"][i]
    path_y = traj["path_y"][i]

    # --- plot desired path FIRST ---
    plt.plot(path_x, path_y, linestyle='--', color='red', linewidth=0.5, label="desired path")

    # --- plot robot trajectory ---
    plt.plot(x[:, i], y[:, i], 'bo-', markersize=1, label="robot")

    # start/end
    plt.scatter(path_x[0], path_y[0], c='k', s=80, label="path start")
    plt.scatter(x[0, i], y[0, i], c='g', s=80, label="robot start")
    plt.scatter(x[-1, i], y[-1, i], c='r', s=80, label="robot end")

    # force pool limits (IMPORTANT)
    plt.xlim(0, 3)
    plt.ylim(0, 2)

    plt.axis("equal")
    plt.grid(True)

    plt.title("Trajectory vs desired path")

    wandb.log({"eval_body/trajectory_vs_path": wandb.Image(traj_fig)})
    plt.close(traj_fig)
    # ---- Heading tracking ----
    fig = plt.figure(figsize=(6,4))
    plt.plot(traj["qh"], label="actual")
    plt.plot(np.unwrap(traj["heading_desired"][:, i]), '--', label="desired")

    plt.title("Heading tracking")
    wandb.log({"eval/heading_tracking": wandb.Image(fig)})
    plt.close(fig)

    # ---- Forward velocity ----
    fig = plt.figure(figsize=(6,4))
    plt.plot(traj["ux"])
    plt.title("Forward velocity")
    wandb.log({"eval/forward_velocity": wandb.Image(fig)})
    plt.close(fig)

    # ---- Lateral velocity ----
    fig = plt.figure(figsize=(6,4))
    plt.plot(traj["uy"])
    plt.title("Lateral velocity")
    wandb.log({"eval/lateral_velocity": wandb.Image(fig)})
    plt.close(fig)

    # ---- Angular velocity ----
    fig = plt.figure(figsize=(6,4))
    plt.plot(traj["qdh"])
    plt.title("Angular velocity")
    wandb.log({"eval/angular_velocity": wandb.Image(fig)})
    plt.close(fig)

    # ---- Control inputs ----
    fig = plt.figure(figsize=(6,4))
    plt.plot(traj["input_alpha"])
    plt.title("Alpha input")
    wandb.log({"eval/input_alpha": wandb.Image(fig)})
    plt.close(fig)

    fig = plt.figure(figsize=(6,4))
    plt.plot(traj["input_delta"])
    plt.title("Delta input")
    wandb.log({"eval/input_delta": wandb.Image(fig)})
    plt.close(fig)

    # ---- Limit cycle ----
    fig = plt.figure(figsize=(5,5))
    plt.plot(traj["tail_u"], traj["qdh"])
    plt.xlabel("tail_u")
    plt.ylabel("qdh")
    plt.title("Limit cycle")
    wandb.log({"eval/limit_cycle": wandb.Image(fig)})
    plt.close(fig)

    #------error plots ------
    fig = plt.figure(figsize=(6,4))
    plt.plot(traj["heading_error"])
    plt.title("Heading error")
    wandb.log({"eval/heading_error": wandb.Image(fig)})
    plt.close(fig)

    fig = plt.figure(figsize=(6,4))
    plt.plot(traj["cross_track_error"])
    plt.title("Cross-track error")
    wandb.log({"eval/cross_track_error": wandb.Image(fig)})
    plt.close(fig)

    print("  -> eval done")


def main(seed=10):

    (
        cfg, eval_cfg, ppo_cfg,
        state, state_eval,
        agent, obs_normalizer,
        graphdef, state_vars,
        rollout_fn,
        eval_rollout_fn,
        key, key_eval,
        N, N_eval, nx, T, num_updates, checkpoint_dir, checkpointer
    ) = init_system(seed)
    start_time = time.time()
    for update in range(num_updates):
        update_start = time.time()

        buffer, state, key, state_vars = collect_rollout(
            rollout_fn, state, key, state_vars, obs_normalizer
        )

        state_vars, metrics, key = update_policy(
            agent, buffer, state,
            obs_normalizer, ppo_cfg,
            graphdef, state_vars, key, cfg
        )

        update_time = time.time() - update_start
        log_metrics(metrics, buffer, update, num_updates, update_time, start_time)

        if update % 100 == 0:

            run_eval_and_log(
                graphdef,
                state_vars,
                obs_normalizer,
                eval_cfg,
                key_eval,
                N_eval,
                nx,
                update,
            )
    final_path = os.path.join(checkpoint_dir, "final")

    checkpointer.save(final_path, state_vars, force=True)
    checkpointer.wait_until_finished()

    obs_normalizer.save(
        os.path.join(final_path, "final_obs_normalizer.npz")
    )

    print(f"✅ Saved final checkpoint to {final_path}")
    print("Training done")

if __name__ == "__main__":
    main(seed=10)
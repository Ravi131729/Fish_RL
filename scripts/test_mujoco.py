"""
Test PPO agent on MuJoCo environments to verify algorithm correctness.
Uses Gymnasium with MuJoCo backend.
"""
print("Importing modules...", flush=True)

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import gymnasium as gym
from typing import Tuple, Optional
import argparse
import wandb

print("Importing local modules...", flush=True)
from networks import Actor, ValueNet, sample_action, compute_log_prob
from ppo_agent import PPO, PPOConfig, RolloutBuffer, compute_gae, ppo_update_with_minibatches
print("Imports done!", flush=True)

class RunningNorm:
    """Running mean/std for observation normalization."""
    def __init__(self, obs_dim, eps=1e-4):
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.var = np.ones(obs_dim, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        # x shape (N, obs_dim)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count

        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

def collect_rollout(
    env,
    model,
    graphdef,
    state_vars,
    rng,
    n_steps,
    obs,
    obs_norm
):

    n_envs = env.num_envs
    obs_dim = obs.shape[-1]

    obs_list = []
    raw_obs_list = []
    actions_list = []
    log_probs_list = []
    values_list = []
    rewards_list = []
    dones_list = []

    # freeze stats for this rollout
    mean = obs_norm.mean.copy()
    var  = obs_norm.var.copy()

    def norm_with_snapshot(x):
        return (x - mean) / np.sqrt(var + 1e-8)

    for _ in range(n_steps):

        raw_obs_list.append(obs.copy())

        obs_n = norm_with_snapshot(obs)
        obs_jax = jnp.array(obs_n)

        rng, action_key = jax.random.split(rng)
        merged = nnx.merge(graphdef, state_vars)

        mean_a, log_std = merged.actor(obs_jax)
        action, log_prob = sample_action(action_key, mean_a, log_std)
        value = merged.critic(obs_jax)

        action_np = np.array(action)

        # store normalized obs (IMPORTANT)
        obs_list.append(obs_n)
        actions_list.append(np.array(action))
        log_probs_list.append(np.array(log_prob))
        values_list.append(np.array(value))

        next_obs, rewards, terminated, truncated, infos = env.step(action_np)
        dones = np.logical_or(terminated, truncated)

        rewards_list.append(rewards)
        dones_list.append(dones.astype(np.float32))

        obs = next_obs   # advance env

    # update running stats AFTER rollout
    raw_all = np.stack(raw_obs_list).reshape(-1, obs_dim)
    obs_norm.update(raw_all)

    buffer = RolloutBuffer(
        obs=jnp.array(np.stack(obs_list)),          # normalized obs
        actions=jnp.array(np.stack(actions_list)),
        log_probs=jnp.array(np.stack(log_probs_list)),
        values=jnp.array(np.stack(values_list)),
        rewards=jnp.array(np.stack(rewards_list)),
        dones=jnp.array(np.stack(dones_list)),
    )

    return buffer, obs, rng



def evaluate_policy(
    env: gym.Env,
    model: PPO,
    graphdef,
    state_vars,
    rng: jax.Array,
    n_episodes: int = 5,
    record_video: bool = False,
    obs_norm: RunningNorm = None,
) -> Tuple[float, Optional[np.ndarray]]:
    """Evaluate policy on single environment without exploration.

    Returns:
        mean_reward: Average episode reward
        video_frames: If record_video=True, returns array of frames (T, H, W, C)
    """

    episode_rewards = []
    video_frames = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # Record frame for first episode only
            if record_video and ep == 0:
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)

            obs_n = obs_norm.normalize(obs)
            obs_jax = jnp.array(obs_n)[None, :]

            merged = nnx.merge(graphdef, state_vars)
            mean, log_std = merged.actor(obs_jax)

            # Use mean action (deterministic) for evaluation
            action = jnp.tanh(mean)
            action_np = np.array(action)[0]

            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)

    frames_array = np.array(video_frames) if video_frames else None
    return float(np.mean(episode_rewards)), frames_array


def train_ppo_mujoco(
    env_name: str = "HalfCheetah-v5",
    n_envs: int = 4,
    total_timesteps: int = 1_000_000,
    n_steps: int = 2048,
    lr_actor: float = 1e-4,   # Reduced from 3e-4 for stable KL
    lr_critic: float = 1e-3,
    seed: int = 42,
    eval_freq: int = 10,
    log_freq: int = 1,
    video_freq: int = 10,
    use_wandb: bool = True,
    wandb_project: str = "ppo-mujoco_walker2d_test",
):
    """Train PPO agent on MuJoCo environment."""

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=wandb_project,
            config={
                "env_name": env_name,
                "n_envs": n_envs,
                "total_timesteps": total_timesteps,
                "n_steps": n_steps,
                "lr_actor": lr_actor,
                "lr_critic": lr_critic,
                "seed": seed,
                "algorithm": "PPO",
            },
            name=f"PPO_{env_name}_{seed}",
        )

    print(f"Training PPO on {env_name}", flush=True)
    print(f"Total timesteps: {total_timesteps}", flush=True)
    print(f"N envs: {n_envs}, N steps per rollout: {n_steps}", flush=True)
    print("-" * 50, flush=True)

    # Create vectorized environment
    print("Creating environments...", flush=True)
    envs = gym.make_vec(env_name, num_envs=n_envs, vectorization_mode="sync")
    eval_env = gym.make(env_name, render_mode="rgb_array")  # For video recording

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    obs_norm = RunningNorm(obs_dim)


    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}", flush=True)

    # Initialize PPO agent
    print("Initializing PPO agent...", flush=True)
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)

    model = PPO(
        obs_dim=obs_dim,
        act_dim=act_dim,
        rng=init_key,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
    )

    # PPO config
    ppo_cfg = PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.0,  # Often 0 for MuJoCo
        max_grad_norm=0.5,
        n_epochs=10,       # Standard for PPO
        n_minibatches=4,   # Fewer minibatches = larger batch per update
        normalize_advantage=True,
        target_kl=0.01,
    )

    # Split model for JIT
    graphdef, state_vars = nnx.split(model)

    # Reset environments
    obs, _ = envs.reset(seed=seed)

    # Training loop
    n_updates = total_timesteps // (n_envs * n_steps)
    timesteps_so_far = 0
    print(f"Starting training for {n_updates} updates...", flush=True)

    for update in range(n_updates):
        # Collect rollout
        rng, rollout_key = jax.random.split(rng)
        buffer, obs, rng = collect_rollout(
            envs, model, graphdef, state_vars, rollout_key, n_steps, obs , obs_norm
        )

        timesteps_so_far += n_envs * n_steps

        # Compute last value for GAE
        obs_n = obs_norm.normalize(obs)
        obs_jax = jnp.array(obs_n)

        merged = nnx.merge(graphdef, state_vars)
        last_value = merged.critic(obs_jax)

        # Compute advantages and returns
        advantages, returns = compute_gae(buffer, last_value, ppo_cfg)

        # Flatten batch
        batch_size = n_steps * n_envs
        obs_flat = buffer.obs.reshape(batch_size, -1)
        act_flat = buffer.actions.reshape(batch_size, -1)
        logp_flat = buffer.log_probs.reshape(batch_size)
        adv_flat = advantages.reshape(batch_size)
        ret_flat = returns.reshape(batch_size)
        val_flat = buffer.values.reshape(batch_size)

        batch_data = (obs_flat, act_flat, logp_flat, adv_flat, ret_flat, val_flat)

        # PPO update
        rng, update_key = jax.random.split(rng)
        state_vars, metrics = ppo_update_with_minibatches(
            graphdef, state_vars, batch_data, update_key, ppo_cfg
        )

        # Logging
        if (update + 1) % log_freq == 0:
            avg_reward = float(buffer.rewards.mean())
            ep_reward_sum = float(buffer.rewards.sum(axis=0).mean())  # Approx episode reward

            print(
                f"Update {update + 1}/{n_updates} | "
                f"Timesteps: {timesteps_so_far} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Policy Loss: {metrics['loss_pi']:.4f} | "
                f"Value Loss: {metrics['loss_v']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f} | "
                f"KL: {metrics['approx_kl']:.4f} | "
                f"Updates: {metrics['n_updates']}",
                flush=True
            )

            # Log to wandb
            if use_wandb:
                wandb.log({
                    "train/avg_reward_per_step": avg_reward,
                    "train/episode_reward_approx": ep_reward_sum,
                    "train/policy_loss": metrics['loss_pi'],
                    "train/value_loss": metrics['loss_v'],
                    "train/entropy": metrics['entropy'],
                    "train/approx_kl": metrics['approx_kl'],
                    "train/clip_fraction": metrics['clipfrac'],
                    "train/n_updates": metrics['n_updates'],
                    "timesteps": timesteps_so_far,
                    "update": update + 1,
                }, step=timesteps_so_far)

        # Evaluation (with optional video recording)
        if (update + 1) % eval_freq == 0:
            record_video = use_wandb and ((update + 1) % video_freq == 0)
            eval_reward, video_frames = evaluate_policy(
                eval_env, model, graphdef, state_vars, rng,
                obs_norm=obs_norm,
                n_episodes=5,
                record_video=record_video
            )

            print(f"  >> Eval reward: {eval_reward:.2f}", flush=True)

            # Log evaluation to wandb
            if use_wandb:
                log_dict = {
                    "eval/mean_reward": eval_reward,
                    "timesteps": timesteps_so_far,
                }

                # Log video if recorded
                if video_frames is not None and len(video_frames) > 0:
                    # wandb expects (T, C, H, W) format, transpose from (T, H, W, C)
                    video_array = np.transpose(video_frames, (0, 3, 1, 2))
                    log_dict["eval/video"] = wandb.Video(video_array, fps=30, format="mp4")
                    print(f"  >> Logged video with {len(video_frames)} frames", flush=True)

                wandb.log(log_dict, step=timesteps_so_far)

    envs.close()
    eval_env.close()

    # Finish wandb run
    if use_wandb:
        wandb.finish()

    print("\nTraining complete!", flush=True)
    return graphdef, state_vars


if __name__ == "__main__":

    graphdef, state_vars = train_ppo_mujoco(
        env_name="Walker2d-v5",
        n_envs=16,
        total_timesteps=3_000_000,
        n_steps=1024,
        lr_actor=3e-4,
        lr_critic=3e-4,
        seed=0,
        eval_freq=10,
        video_freq=10,
        use_wandb=True,
    )

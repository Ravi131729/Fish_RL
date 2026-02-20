import jax
import jax.numpy as jnp
from flax import nnx
import wandb
import time
import os
import orbax.checkpoint as ocp
import numpy as np

from fish.env.fish_env import EnvConfig, reset_env, build_obs, step_env, compute_reward,head_position,eval_step_env
from fish.agents.networks import sample_action
from fish.agents.ppo_agent import (
    PPO,
    RolloutBuffer,
    PPOConfig,
    compute_gae,
    ppo_update_step,
    ppo_update_with_minibatches,
)
from fish.utils.obs_normalizer import RunningMeanStd
import matplotlib.pyplot as plt
import matplotlib
from fish.utils.load_checkpoint import load_agent
matplotlib.use("Agg")

def make_eval_rollout(graphdef, cfg: EnvConfig, T: int):
    """
    Deterministic evaluation rollout (no sampling).
    Use small N (e.g., 10 envs).
    Returns trajectories for plotting/logging.
    """

    @jax.jit
    def eval_rollout_fn(state, key, state_vars_in, obs_mean, obs_std):

        agent = nnx.merge(graphdef, state_vars_in)

        def normalize_obs(obs):
            return jnp.clip((obs - obs_mean) / obs_std, -10.0, 10.0)

        def one_step(carry, _):
            env_state, key_step = carry
            key_step, k_env = jax.random.split(key_step)

            obs_raw = build_obs(env_state,cfg,key)
            obs_norm = normalize_obs(obs_raw)


            mean, log_std = agent.actor(obs_norm)
            actions = jnp.tanh(mean)              # use mean directly
            values = agent.critic(obs_norm)

            delta_raw = actions[:, 0]

            env_state_next = eval_step_env(env_state, delta_raw, k_env, cfg)

            rewards, info = compute_reward(
                env_state, env_state_next, delta_raw, cfg
            )

            # extract head position (adjust if different)
            xpos, ypos = head_position(env_state_next)

            # ux_desired = env_state_next.ux_desired
            heading_desired = env_state_next.heading_desired

            path_x = env_state_next.paths[:, :, 0]
            path_y = env_state_next.paths[:, :, 1]


            data = (
                xpos, ypos,
                info["ux_avg"], info["uy_avg"],
                info["heading_avg"],
                info["omega_avg"],
                env_state_next.alpha_prev,
                env_state_next.delta_prev,
                rewards,
                info["u"],
                info["qh"],
                info["qdh"],
                info["u_x"],
                info["u_y"],


                # ux_desired,
                info["heading_error"],
                heading_desired,
                delta_raw,
                info["cross_track_error"],
                path_x,
                path_y,



            )

            return (env_state_next, key_step), data

        (state_T, key_out), traj = jax.lax.scan(
            one_step, (state, key), None, length=T
        )

        xpos, ypos, ux_avg, uy_avg, heading, omega, alpha_raw, delta_raw, rewards, u, qh, qhdot,u_x,u_y,heading_error,heading_desired,delta_raw,cross_track_error,path_x,path_y    = traj

        trajectories = {
            "x": xpos,
            "y": ypos,
            "ux": ux_avg,

            "uy": uy_avg,
            "heading": heading,
            "omega": omega,
            "input_alpha": alpha_raw,
            "input_delta": delta_raw,
            "delta_raw": delta_raw,
            "reward": rewards,
            "qh": qh,
            "tail_u": u,
            "qdh": qhdot,
            "u_x": u_x,
            "u_y": u_y,

            "heading_error": heading_error,

            "heading_desired": heading_desired,
            "cross_track_error": cross_track_error,
            "path_x": path_x,
            "path_y": path_y,

        }

        return trajectories, state_T, key_out

    return eval_rollout_fn


def make_jitted_rollout(graphdef, state_vars, cfg: EnvConfig, T: int, N: int):
    """Return a jitted rollout function that operates on nnx graph/state.

    This avoids putting the nnx.Module itself (`agent`) into the jitted
    function. Instead we:
      * merge `graphdef` + `state_vars` inside the jitted body to get an agent
      * use lax.scan for the temporal loop

    The function accepts obs_mean/obs_std arrays so that observations are
    normalized before being fed to the actor and critic.  Raw (un-normalized)
    observations are still stored in the buffer for running-stats updates.

    Returned function signature:
        (state, key, state_vars, obs_mean, obs_std)
            -> (buffer, state_T, key, new_state_vars)
    """

    @jax.jit
    def rollout_fn(state, key, state_vars_in, obs_mean, obs_std):
        # Reconstruct the agent module from graphdef + state vars
        agent = nnx.merge(graphdef, state_vars_in)

        def normalize_obs(obs):
            return jnp.clip((obs - obs_mean) / obs_std, -10.0, 10.0)

        def one_step(carry, _):
            env_state, key_step = carry
            key_step, k_act, k_env = jax.random.split(key_step, 3)

            obs_raw = build_obs(env_state,cfg,key)  # (N, obs_dim)
            obs_norm = normalize_obs(obs_raw)

            # policy action and value on NORMALIZED obs
            mean, log_std = agent.actor(obs_norm)
            actions, logp = sample_action(k_act, mean, log_std)
            values = agent.critic(obs_norm)

            delta_raw = actions[:, 0]  # (N,) - extract scalar action from (N, act_dim)

            env_state_next = step_env(env_state, delta_raw, k_env, cfg)

            rewards, info = compute_reward(env_state, env_state_next,delta_raw, cfg)


            # Store raw obs for running-stats update; normalized obs not needed
            data = (obs_raw, actions, logp, values, rewards, env_state_next.done,info["ux_avg"],info["uy_avg"],info["qh"],info["heading_avg"],info["omega_avg"],info["u"],info["qdh"],info["u_x"],info["u_y"],info["heading_error"],info["cross_track_error"])
            return (env_state_next, key_step), data

        (state_T, key_out), (obs, act, logp, val, rew, done,ux_avg,uy_avg,qh,heading_avg,omega_avg,u,qhdot,u_x,u_y,heading_error,cross_track_error) = jax.lax.scan(
            one_step, (state, key), None, length=T
        )

        buffer = RolloutBuffer(
            obs=obs,
            actions=act,
            log_probs=logp,
            values=val,
            rewards=rew,
            dones=done,
            avg_forward_velocity = ux_avg,
            avg_lateral_velocity = uy_avg,
            avg_heading = heading_avg,
            qh = qh,
            omega_avg = omega_avg,
            u = u,
            qdh = qhdot,
            u_x = u_x,
            u_y = u_y,
            # velocity_error = velocity_error,
            heading_error = heading_error,
            cross_track_error = cross_track_error
        )

        # Split back out updated module state for the caller
        _, new_state_vars = nnx.split(agent)
        return buffer, state_T, key_out, new_state_vars

    return rollout_fn


def main(seed=10):
    """Dummy PPO training loop: rollout + a few PPO updates."""
    # RNG setup
    cfg = EnvConfig(
        dt=0.01,
        max_steps=500,
        delta_max=1.0,           # servo max angle (radians)
        delta_rate_max=5.0,      # servo max angular velocity (radians/sec)

        alpha_max=4619,    #A*w**2 max = 13*2pi*3^2 = 4619
        alpha_rate_max=87000,  #A*w**3 max = 13*2pi*3^3 ~ 87000,
        max_heading_angle=jnp.pi,  # max heading error for reward (radians)
        min_heading_angle=-jnp.pi,  # min heading error for reward (radians)
        max_ux = 0.5,
        min_ux = 0.05,
        omega_rotor_max = 245, # max A*w^3 = 13*2pi*3^3 ~ 87000
        A_max = 13, # max A in A*w^3 = 13*2pi*3^3 ~ 87000
        A_min = 6, # min A to avoid singularity in reward when A*w^3 is near zero
        w_max = 3.0, # max w in A*w^3 = 13*2pi*3^3 ~ 87000
        w_min = 2.0, # min w to avoid singularity in reward when A*w^3 is near zero


        beta=0.02,
        max_position=1,

    )
    eval_cfg = cfg.replace(max_steps=20000, dt=0.001)  # copy base config

    ppo_cfg = PPOConfig()

    # Parallel envs
    N = 512
    N_eval = 1
    nx = 7
    # Training hyperparams (small values for a quick test)
    T = 256 # horizon
    num_updates = 1000
    # Initialize wandb
    wandb.init(
        project="fish_path_tracking_lookaheadpoint",
        config={
            "seed": seed,
            "env": {
                "dt": cfg.dt,
                "max_steps": cfg.max_steps,
                "beta": cfg.beta,
                "alpha_max": cfg.alpha_max,
                "alpha_rate_max": cfg.alpha_rate_max,
                "delta_max": cfg.delta_max,
                "delta_rate_max": cfg.delta_rate_max,
                "max_heading_angle": cfg.max_heading_angle,
                "min_heading_angle": cfg.min_heading_angle,
                "max_ux": cfg.max_ux,
                "min_ux": cfg.min_ux,
            },
            "ppo": {
                "gamma": ppo_cfg.gamma,
                "gae_lambda": ppo_cfg.gae_lambda,
                "clip_eps": ppo_cfg.clip_eps,
                "vf_coef": ppo_cfg.vf_coef,
                "ent_coef": ppo_cfg.ent_coef,
                "n_epochs": ppo_cfg.n_epochs,
                "n_minibatches": ppo_cfg.n_minibatches,
                "target_kl": ppo_cfg.target_kl,
            },
            "training": {
                "n_envs": N,
                "horizon": T,
                "num_updates": num_updates,

            },
        },
    )
    wandb.save("*.py")
    seed = wandb.config.seed
    print("Running seed:", seed)
    key = jax.random.PRNGKey(seed)
    key, key_env, key_agent, key_eval = jax.random.split(key, 4)

    # Env config (small values for a quick smoke test)

    state = reset_env(key_env, N, nx, cfg=cfg)
    state_eval = reset_env(key_env, N_eval, nx, cfg=eval_cfg)


    # Build one observation to get obs_dim
    obs0 = build_obs(state,cfg,key)  # (N, obs_dim)
    obs_dim = obs0.shape[1]
    act_dim =1 # delta_raw is scalar per env

    # PPO agent + config
    agent = PPO(obs_dim, act_dim, rng=key_agent)

    # Observation normalizer (running mean/std)
    obs_normalizer = RunningMeanStd(obs_dim)

    # Load actor and normalizer from checkpoint (if available)
    # agent, obs_normalizer = load_agent("final", obs_dim, act_dim)



    print("Dummy PPO train: starting...")

    # Prepare nnx graph/state for ppo_update_step and jitted rollout
    graphdef, state_vars = nnx.split(agent)
    rollout_fn = make_jitted_rollout(graphdef, state_vars, cfg, T, N)

    # Setup checkpointing
    checkpoint_dir = os.path.abspath("./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    save_every = 300 # Save checkpoint every N updates
    best_reward = float('-inf')

    # Time tracking
    start_time = time.time()

    for update in range(num_updates):
        update_start = time.time()

        # Get current normalizer stats as JAX arrays for the jitted rollout
        obs_mean = jnp.array(obs_normalizer.mean, dtype=jnp.float32)
        obs_std = jnp.sqrt(jnp.array(obs_normalizer.var, dtype=jnp.float32) + 1e-8)

        # 1) Collect rollout (jitted) — actor/critic see normalized obs
        buffer, state, key, state_vars = rollout_fn(
            state, key, state_vars, obs_mean, obs_std
        )

        # 1b) Update running obs stats from raw observations in the buffer
        obs_normalizer.update(np.asarray(buffer.obs))  # (T, N, obs_dim)

        # 2) Compute bootstrap value at last state (with normalized obs)
        last_obs_raw = build_obs(state,cfg,key)  # (N, obs_dim)
        last_obs_norm = obs_normalizer.normalize(last_obs_raw)
        last_value = agent.critic(last_obs_norm)  # (N,)

        # 3) GAE advantages and returns
        advantages, returns = compute_gae(buffer, last_value, ppo_cfg)

        # Flatten (T, N) -> (T*N,)
        T_, N_ = advantages.shape
        # Normalize the raw obs for PPO update
        obs_norm_flat = obs_normalizer.normalize(
            buffer.obs.reshape(T_ * N_, -1)
        )
        act_flat = buffer.actions.reshape(T_ * N_, -1)
        logp_flat = buffer.log_probs.reshape(T_ * N_)
        val_flat = buffer.values.reshape(T_ * N_)
        adv_flat = advantages.reshape(T_ * N_)
        ret_flat = returns.reshape(T_ * N_)

        # 4) PPO update with minibatches and multiple epochs
        batch_data = (obs_norm_flat, act_flat, logp_flat, adv_flat, ret_flat, val_flat)
        key, key_update = jax.random.split(key)
        state_vars, metrics = ppo_update_with_minibatches(
            graphdef,
            state_vars,
            batch_data,
            key_update,
            ppo_cfg,
        )
# dist_error = dist_err,
#             dist_next_error = dist_next_err,
#             progress = progress,
#             heading_error = e_psi
        # Track rewards
        ep_reward_mean = float(buffer.rewards.sum(axis=0).mean())  # mean episode reward across envs
        ep_reward_std = float(buffer.rewards.sum(axis=0).std())
        reward_per_step = float(buffer.rewards.mean())
        avg_forward_velocity_mean = float(buffer.avg_forward_velocity.mean())
        avg_lateral_velocity_mean = float(buffer.avg_lateral_velocity.mean())
        qh_mean = float(buffer.qh.mean())
        avg_heading_mean = float(buffer.avg_heading.mean())
        omega_avg_mean = float(buffer.omega_avg.mean())

        # Time tracking
        update_time = time.time() - update_start
        elapsed = time.time() - start_time
        remaining = (num_updates - update - 1) * (elapsed / (update + 1))

        # Log to wandb
        wandb.log({
            "train/loss_total": float(metrics['loss_total']),
            "train/loss_pi": float(metrics['loss_pi']),
            "train/loss_v": float(metrics['loss_v']),
            "train/entropy": float(metrics['entropy']),
            "train/approx_kl": float(metrics['approx_kl']),
            "train/clipfrac": float(metrics['clipfrac']),
            "train/n_updates": metrics['n_updates'],
            "train/early_stopped": int(metrics['early_stopped']),
            "reward/ep_reward_mean": ep_reward_mean,
            "reward/ep_reward_std": ep_reward_std,
            "reward/reward_per_step": reward_per_step,
            "time/update_time": update_time,
            "time/elapsed": elapsed,
            "state_stats/avg_forward_velocity_mean": avg_forward_velocity_mean,
            "state_stats/avg_lateral_velocity_mean": avg_lateral_velocity_mean,
            "state_stats/qh_mean": qh_mean,
            "state_stats/avg_heading_mean": avg_heading_mean,
            "state_stats/omega_avg_mean": omega_avg_mean,
            # "state_stats/velocity_error_mean": float(buffer.velocity_error.mean()),
            "state_stats/heading_error_mean": float(buffer.heading_error.mean()),
            "state_stats/cross_track_error_mean": float(buffer.cross_track_error.mean()),
            "update": update + 1,
        })

        if (update + 1) % 10 == 0 or update == 0:
            print(
                f"update {update + 1}/{num_updates}: "
                f"loss={float(metrics['loss_total']):.4f}, "
                f"loss_pi={float(metrics['loss_pi']):.4f}, "
                f"loss_v={float(metrics['loss_v']):.4f}, "
                f"entropy={float(metrics['entropy']):.4f}, "
                f"approx_kl={float(metrics['approx_kl']):.4f}, "
                f"ep_rew={ep_reward_mean:.2f}±{ep_reward_std:.2f}, "
                f"rew/step={reward_per_step:.4f}, "
                f"ETA={remaining/60:.1f}min",

            )
        if (update + 1) % save_every == 0:
            print(f"  -> running eval {update + 1}")
            state_eval = reset_env(key_eval, N_eval, nx, eval_cfg)
            eval_fn = make_eval_rollout(graphdef, eval_cfg, T=eval_cfg.max_steps-1)

            traj, state_eval, key = eval_fn(
                state_eval, key_eval,
                state_vars,
                obs_mean,
                obs_std
            )
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


            inp_fig = plt.figure(figsize=(6,4))
            plt.plot(traj["input_alpha"])
            plt.title("Eval input alpha")
            wandb.log({"eval_body/input_alpha": wandb.Image(inp_fig)})
            plt.close(inp_fig)
            ux_fig = plt.figure(figsize=(6,4))
            plt.plot(traj["ux"])
            plt.title("Eval avg forward velocity")
            wandb.log({"eval_avg/avg_forward_velocity": wandb.Image(ux_fig)})
            plt.close(ux_fig)
            uy_fig = plt.figure(figsize=(6,4))
            plt.plot(traj["uy"])
            plt.title("Eval avg lateral velocity")
            wandb.log({"eval_avg/avg_lateral_velocity": wandb.Image(uy_fig)})
            plt.close(uy_fig)
            omega_fig = plt.figure(figsize=(6,4))
            plt.plot(traj["omega"])
            plt.title("Eval avg angular velocity")
            wandb.log({"eval_avg/avg_angular_velocity": wandb.Image(omega_fig)})
            plt.close(omega_fig)
            heading_fig = plt.figure(figsize=(6,4))
            plt.plot(traj["heading"])
            plt.title("Eval avg heading")
            wandb.log({"eval_avg/avg_heading": wandb.Image(heading_fig)})
            plt.close(heading_fig)
            tail_u_fig = plt.figure(figsize=(6,4))
            plt.plot(traj["tail_u"],traj["qdh"])
            plt.title("limit cycle: tail_u vs qhdot")
            wandb.log({"eval_body/tail_u_vs_qhdot": wandb.Image(tail_u_fig)})
            plt.close(tail_u_fig)
            ubx_fig = plt.figure(figsize=(6,4))
            plt.plot(traj["u_x"])
            # plt.plot(traj["ux_desired"], color='r', linestyle='--')
            plt.title("Eval body forward velocity")
            wandb.log({"eval_body/body_forward_velocity": wandb.Image(ubx_fig)})
            plt.close(ubx_fig)
            uby_fig = plt.figure(figsize=(6,4))
            plt.plot(traj["u_y"])
            plt.title("Eval body lateral velocity")
            wandb.log({"eval_body/body_lateral_velocity": wandb.Image(uby_fig)})
            plt.close(uby_fig)


            act_heading_fig = plt.figure(figsize=(6,4))
            heading_des = np.unwrap(traj["heading_desired"][:, i])
            heading_fig = plt.figure(figsize=(6,4))
            plt.plot(traj["qh"][:, i], label="actual heading")
            plt.plot(heading_des, '--', label="desired heading")
            plt.title("Eval qh (heading)")
            wandb.log({"eval_body/qh": wandb.Image(heading_fig)})
            plt.close(heading_fig)

            act_omega_fig = plt.figure(figsize=(6,4))
            plt.plot(traj["qdh"])
            plt.title("Eval omega (body angular velocity)")
            wandb.log({"eval_body/omega": wandb.Image(act_omega_fig)})
            plt.close(act_omega_fig)
            delta_fig = plt.figure(figsize=(6,4))
            plt.plot(traj["input_delta"])
            plt.title("Eval input delta")
            wandb.log({"eval_body/input_delta": wandb.Image(delta_fig)})
            plt.close(delta_fig)


            print(f"  ->  eval done")

        # Save checkpoint periodically
        if (update + 1) % save_every == 0:
            # Reconstruct agent with updated state for saving
            agent_to_save = nnx.merge(graphdef, state_vars)
            _, ckpt_state = nnx.split(agent_to_save)

            ckpt_path = os.path.join(checkpoint_dir, f"step_{update + 1}")
            checkpointer.save(ckpt_path, ckpt_state, force=True)
            checkpointer.wait_until_finished()
            # Save normalizer alongside checkpoint
            obs_normalizer.save(ckpt_path + "_obs_normalizer.npz")
            print(f"  -> Saved checkpoint to {ckpt_path}")

        # Save best model based on reward
        if ep_reward_mean > best_reward:
            best_reward = ep_reward_mean
            agent_to_save = nnx.merge(graphdef, state_vars)
            _, ckpt_state = nnx.split(agent_to_save)

            best_path = os.path.join(checkpoint_dir, "best")
            checkpointer.save(best_path, ckpt_state, force=True)
            checkpointer.wait_until_finished()
            obs_normalizer.save(os.path.join(checkpoint_dir, "best_obs_normalizer.npz"))
    ################################################################
    N_final_evals =3

    for i_eval in range(N_final_evals):

        print(f"\nRunning FINAL eval {i_eval}")

        # reset eval env
        key_eval, subkey = jax.random.split(key_eval)
        state_eval = reset_env(subkey, N_eval, nx, eval_cfg)

        eval_fn = make_eval_rollout(graphdef, eval_cfg, T=eval_cfg.max_steps-1)

        traj, state_eval, key_eval = eval_fn(
            state_eval, key_eval,
            state_vars,
            obs_mean,
            obs_std
        )
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

        wandb.log({f"final_eval/trajectory_{i_eval}": wandb.Image(traj_fig)})
        plt.close(traj_fig)

        # ---------------- alpha input ----------------
        fig = plt.figure()
        plt.plot(traj["input_alpha"])
        plt.title(f"alpha final eval {i_eval}")
        wandb.log({f"final_eval/input_alpha_{i_eval}": wandb.Image(fig)})
        plt.close(fig)

        # ---------------- forward velocity ----------------
        fig = plt.figure()
        plt.plot(traj["u_x"])
        # plt.plot(traj["ux_desired"], color='r', linestyle='--')
        plt.title(f"body forward vel {i_eval}")
        wandb.log({f"final_eval/body_forward_vel_{i_eval}": wandb.Image(fig)})
        plt.close(fig)

        # ---------------- heading ----------------
        fig = plt.figure()
        plt.plot(traj["qh"])
        plt.plot(traj["heading_desired"], color='r', linestyle='--')
        plt.title(f"heading final eval {i_eval}")
        wandb.log({f"final_eval/heading_{i_eval}": wandb.Image(fig)})
        plt.close(fig)

        # ---------------- delta ----------------
        fig = plt.figure()
        plt.plot(traj["input_delta"])
        plt.title(f"delta final eval {i_eval}")
        wandb.log({f"final_eval/input_delta_{i_eval}": wandb.Image(fig)})
        plt.close(fig)


    print("\nAll final evals done.")
    ##############################################
    # Save final checkpoint

    agent_final = nnx.merge(graphdef, state_vars)
    _, final_state = nnx.split(agent_final)
    final_path = os.path.join(checkpoint_dir, "final")
    checkpointer.save(final_path, final_state, force=True)
    checkpointer.wait_until_finished()
    obs_normalizer.save(os.path.join(checkpoint_dir, "final_obs_normalizer.npz"))
    print(f"Saved final checkpoint to {final_path}")

    total_time = time.time() - start_time
    print(f"Dummy PPO train: finished in {total_time/60:.1f} minutes.")
    print(f"Best episode reward: {best_reward:.2f}")
    wandb.finish()


if __name__ == "__main__":
    main()

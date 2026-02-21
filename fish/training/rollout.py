import jax
import jax.numpy as jnp
from flax import nnx

from fish.env.env_fish import step_env
from fish.env.observation import build_obs
from fish.env.reward import compute_reward
from fish.env.action_parser import parse_action

from fish.agents.networks import sample_action
from fish.agents.ppo_agent import RolloutBuffer


def make_jitted_rollout(graphdef, state_vars, cfg, T: int, N: int):

    @jax.jit
    def rollout_fn(state, key, state_vars_in, obs_mean, obs_std):

        agent = nnx.merge(graphdef, state_vars_in)

        # --------------------------------------------------
        # normalize obs
        # --------------------------------------------------
        def normalize_obs(obs):
            return jnp.clip((obs - obs_mean) / obs_std, -10.0, 10.0)

        # --------------------------------------------------
        # policy forward
        # --------------------------------------------------
        def policy_forward(obs_norm, key_act):
            mean, log_std = agent.actor(obs_norm)
            actions, logp = sample_action(key_act, mean, log_std)
            values = agent.critic(obs_norm)
            return actions, logp, values

        # --------------------------------------------------
        # single step
        # --------------------------------------------------
        def one_step(carry, _):
            env_state, key_step = carry
            key_step, k_act, k_env = jax.random.split(key_step, 3)

            # ---- obs ----
            obs_raw = build_obs(env_state, cfg, key_step)
            obs_norm = normalize_obs(obs_raw)

            # ---- policy ----
            action_vec, logp, values = policy_forward(obs_norm, k_act)

            # ---- parse action ----
            action = parse_action(action_vec, env_state, cfg)

            # ---- env step ----
            env_state_next = step_env(env_state, action, k_env, cfg)

            rewards, info = compute_reward(
                env_state, env_state_next, action, cfg
            )

            step_data = {
                "obs": obs_raw,
                "actions": action_vec,
                "logp": logp,
                "values": values,
                "rewards": rewards,
                "done": env_state_next.done,

                # logging
                "ux_avg": info["ux_avg"],
                "uy_avg": info["uy_avg"],
                "heading_avg": info["heading_avg"],
                "qh": info["qh"],
                "omega_avg": info["omega_avg"],
                "u": info["u"],
                "qdh": info["qdh"],
                "ux": info["ux"],
                "uy": info["uy"],
                "heading_error": info["heading_error"],
                "cross_track_error": info["cross_track_error"],
            }

            return (env_state_next, key_step), step_data

        # --------------------------------------------------
        # scan rollout
        # --------------------------------------------------
        (state_T, key_out), traj = jax.lax.scan(
            one_step, (state, key), None, length=T
        )

        # --------------------------------------------------
        # PPO buffer
        # --------------------------------------------------
        buffer = RolloutBuffer(
            obs=traj["obs"],
            actions=traj["actions"],
            log_probs=traj["logp"],
            values=traj["values"],
            rewards=traj["rewards"],
            dones=traj["done"],
            avg_forward_velocity=traj["ux_avg"],
            avg_lateral_velocity=traj["uy_avg"],
            avg_heading=traj["heading_avg"],
            qh=traj["qh"],
            omega_avg=traj["omega_avg"],
            u=traj["u"],
            qdh=traj["qdh"],
            ux=traj["ux"],
            uy=traj["uy"],
            heading_error=traj["heading_error"],
            cross_track_error=traj["cross_track_error"],
        )

        # updated nnx state
        _, new_state_vars = nnx.split(agent)

        return buffer, state_T, key_out, new_state_vars

    return rollout_fn
import jax
import jax.numpy as jnp
from flax import nnx

from fish.env.env_fish import eval_step_env
from fish.env.observation import build_obs
from fish.env.reward import compute_reward
from fish.env.action_parser import parse_action
from fish.env.kinematics import head_position


def make_eval_rollout(graphdef, cfg, T: int):

    @jax.jit
    def eval_rollout_fn(state, key, state_vars_in, obs_mean, obs_std):

        agent = nnx.merge(graphdef, state_vars_in)

        # --------------------------------------------------
        # normalize
        # --------------------------------------------------
        def normalize_obs(obs):
            return jnp.clip((obs - obs_mean) / obs_std, -10.0, 10.0)

        # --------------------------------------------------
        # deterministic policy
        # --------------------------------------------------
        def policy_forward(obs_norm):
            mean, log_std = agent.actor(obs_norm)
            actions = jnp.tanh(mean)   # deterministic
            values = agent.critic(obs_norm)
            return actions, values

        # --------------------------------------------------
        # one step
        # --------------------------------------------------
        def one_step(carry, _):
            env_state, key_step = carry
            key_step, k_env = jax.random.split(key_step)

            # --- obs ---
            obs_raw = build_obs(env_state, cfg, key)
            obs_norm = normalize_obs(obs_raw)

            # --- policy ---
            action_vec, values = policy_forward(obs_norm)

            action = parse_action(action_vec, env_state, cfg)

            env_state_next = eval_step_env(env_state, action, k_env, cfg)

            rewards, info = compute_reward(
                env_state, env_state_next, action, cfg
            )

            xpos, ypos = head_position(env_state_next)

            step_data = {
                "x": xpos,
                "y": ypos,
                "ux_avg": info["ux_avg"],
                "uy_avg": info["uy_avg"],
                "heading_avg": info["heading_avg"],
                "omega_avg": info["omega_avg"],
                "input_alpha": env_state_next.alpha_prev,
                "input_delta": env_state_next.delta_prev,
                "reward": rewards,
                "tail_u": info["u"],
                "qh": info["qh"],
                "qdh": info["qdh"],
                "ux": info["ux"],
                "uy": info["uy"],
                "heading_error": info["heading_error"],
                "heading_desired": env_state_next.heading_desired,
                "cross_track_error": info["cross_track_error"],
                "path_x": env_state_next.paths[:, :, 0],
                "path_y": env_state_next.paths[:, :, 1],
                "action_raw": action_vec,
            }

            return (env_state_next, key_step), step_data

        # --------------------------------------------------
        # rollout scan
        # --------------------------------------------------
        (state_T, key_out), traj = jax.lax.scan(
            one_step, (state, key), None, length=T
        )

        return traj, state_T, key_out

    return eval_rollout_fn
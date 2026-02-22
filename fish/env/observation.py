import jax.numpy as jnp
import jax
from fish.env.kinematics import head_position,world_velocity, body_velocity
from fish.utils.path_utils import compute_path_errors,circle_lookahead


def add_obs_noise(key, obs, sigma=0.01):
    noise = sigma * jax.random.normal(key, obs.shape)
    return obs + noise


def build_obs(state, cfg, key=None):

    xpos, ypos = head_position(state)
    qh = state.x[:,2]
    vx,vy = world_velocity(
        xpos,
        ypos,
        state.head_x_prev,
        state.head_y_prev,
        cfg.dt
    )
    ux,uy = body_velocity(vx, vy, qh)

    ct_err, hd_err, path_heading, idx = compute_path_errors(
        state.paths,
        xpos,
        ypos,
        qh,
    )
    target, heading_des, new_idx, found = circle_lookahead(
        state.paths,
        xpos,
        ypos,
        state.path_idx,
        L=0.25
    )

    hd_err  = qh - heading_des
    hd_err = jnp.arctan2(jnp.sin(hd_err), jnp.cos(hd_err))

    obs = jnp.concatenate([
        ux[:,None],
        # uy[:,None],
        # qh[:,None],
        # ct_err[:,None],
        
        hd_err[:,None],
        state.delta_prev[:,None],
    ], axis=1)

    if key is not None:
        obs = add_obs_noise(key, obs)

    return obs
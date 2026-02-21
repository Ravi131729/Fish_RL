import jax.numpy as jnp
from fish.env.kinematics import head_position, world_velocity, body_velocity
from fish.utils.path_utils import compute_path_errors


def compute_reward(state, state_next, action, cfg):

    xpos, ypos = head_position(state)
    qh = state.x[:,2]
    u  = state.x[:,3]
    qdh = state.x[:, -1]
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

    # normalize heading
    hd = hd_err / jnp.pi

    delta_change = state_next.delta_prev - state.delta_prev

    reward = (
        - 2.0 * ct_err**2
        - 0.5 * hd**2
        - 0.01 * delta_change**2
    )

    reward = jnp.clip(reward, -3.0, 3.0)


    info = {
    "ux_avg": state.ux_avg,
    "uy_avg": state.uy_avg,
    "heading_avg": state.heading_avg,
    "omega_avg": state.omega_avg,
    "qh": qh,
    "u": u,
    "qdh": qdh,
    "ux": ux,
    "uy": uy,
    "heading_error": hd_err,
    "cross_track_error": ct_err,


    }

    return reward, info
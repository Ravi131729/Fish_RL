import jax.numpy as jnp
from fish.env.kinematics import head_position, world_velocity, body_velocity
from fish.utils.path_utils import compute_path_errors,circle_lookahead


def compute_reward(state, state_next, action, cfg):
    kp_raw = action["kp"]   # [-1,1]
    kd_raw = action["kd"]   # [-1,1]

    kp_min, kp_max = 0.0, 10.0
    kd_min, kd_max = 0.0, 1.0

    kp = kp_min + 0.5*(kp_raw + 1.0) * (kp_max - kp_min)
    kd = kd_min + 0.5*(kd_raw + 1.0) * (kd_max - kd_min)

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
    target, heading_des, new_idx, found = circle_lookahead(
        state.paths,
        xpos,
        ypos,
        state.path_idx,
        L=0.25
    )

    hd_err  = qh - heading_des
    hd_err = jnp.arctan2(jnp.sin(hd_err), jnp.cos(hd_err))

    # normalize heading
    hd = hd_err / jnp.pi

    delta_change = state_next.delta_prev - state.delta_prev

    reward = (

        - 0.5 * hd**2
        # - 0.01 * delta_change**2
        - 0.001 * kp**2
        - 0.01 * kd**2
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
    "look_ahead_point_x": target[:,0],
    "look_ahead_point_y": target[:,1],


    }

    return reward, info
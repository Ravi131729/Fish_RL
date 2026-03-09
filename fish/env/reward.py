import jax.numpy as jnp
from fish.env.kinematics import head_position, world_velocity, body_velocity
from fish.utils.path_utils import compute_path_errors,circle_lookahead
from fish.env.action_parser import get_pid_gains


def compute_reward(state, state_next, action, cfg):
    kp, kd, ki, v_kp, v_kd, v_ki = get_pid_gains(action, cfg)

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
        L=state.L,
    )

    hd_err  = qh - heading_des
    hd_err = jnp.arctan2(jnp.sin(hd_err), jnp.cos(hd_err))
    speed_err = ux - state.desired_ux
    speed_err = speed_err /cfg.max_ux

    # normalize heading
    hd = hd_err / jnp.pi

    delta_change = state_next.delta_prev - state.delta_prev

    reward = (
        - 2.0 * speed_err**2

        - 1.0* hd**2
        # - 0.01 * delta_change**2
        - 0.001* kp**2
        - 0.01 * kd**2
        -0.01 * ki**2

        - 0.001 * v_kp**2
        - 0.01 * v_kd**2
        - 0.01 * v_ki**2

    )

    reward = jnp.clip(reward, -5.0, 5.0)


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
    "speed_error": speed_err,
    "look_ahead_point_x": target[:,0],
    "look_ahead_point_y": target[:,1],
    "desired_heading": heading_des,
    "desired_ux": state.desired_ux,
    "kp": kp,
    "kd": kd,
    "ki": ki,
    "v_kp": v_kp,
    "v_kd": v_kd,
    "v_ki": v_ki,
    "L": state.L
    }

    return reward, info
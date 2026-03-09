from fish.env.dynamics import step as dynamics_step
from fish.env.kinematics import (
    head_position,
    world_velocity,
    body_velocity,
    update_tail_position,
)
import jax
import jax.numpy as jnp
from fish.env.types import EnvState, EnvConfig
from fish.utils.path_utils import compute_path_errors,circle_lookahead
from fish.env.reset import sample_physics_params, reset_env , compute_done
from fish.env.kinematics import head_position ,world_velocity, body_velocity
from fish.env.action_parser import get_pid_gains

def make_input(t, alpha,delta):


    return jnp.stack(
        [delta,
         jnp.zeros_like(delta),
         jnp.zeros_like(delta),
         alpha],
        axis=1,
    )
import jax.numpy as jnp

import jax
import jax.numpy as jnp
@jax.jit
def pick_A_omega_from_S_linear_batched(S,cfg):
    """
    S: shape (N_env,)
    Returns:
        A: shape (N_env,)
        w: shape (N_env,)
    """

    # Build grid once
    A_vals = jnp.linspace(cfg.A_min, cfg.A_max, 20)              # (nA,)
    omega_vals = jnp.linspace(2*jnp.pi*cfg.w_min, 2*jnp.pi*cfg.w_max, 10)              # (nw,)

    A_mesh, omega_mesh = jnp.meshgrid(A_vals, omega_vals, indexing="ij")  # (nA, nw)

    # Compute grid S
    S_grid = (A_mesh * (omega_mesh**2) / cfg.alpha_max)**2          # (nA, nw)

    # Flatten grid
    S_flat = S_grid.reshape(-1)                        # (nA*nw,)
    A_flat = A_mesh.reshape(-1)
    omega_flat = omega_mesh.reshape(-1)

    # Expand S for broadcasting
    # S: (N_env,) -> (N_env, 1)
    S_exp = S[:, None]

    # Compute absolute difference to every grid point
    diff = jnp.abs(S_exp - S_flat[None, :])            # (N_env, nA*nw)

    # Argmin per environment
    idx = jnp.argmin(diff, axis=1)                     # (N_env,)

    # Gather
    A_out = A_flat[idx]
    omega_out = omega_flat[idx]

    return A_out, omega_out

@jax.jit
def step_core(state: EnvState, action, cfg: EnvConfig):

    kp, kd, ki, v_kp, v_kd, v_ki = get_pid_gains(action, cfg)
    # L = get_lookahead(action, cfg)
    xpos, ypos = head_position(state)
    qh = state.x[:, 2]
    vx,vy = world_velocity(
        xpos,
        ypos,
        state.head_x_prev,
        state.head_y_prev,
        cfg.dt
    )
    ux,uy = body_velocity(vx, vy, qh)

    hd_error = qh - state.heading_desired
    hd_error = jnp.arctan2(jnp.sin(hd_error), jnp.cos(hd_error))
    hd_error = hd_error/jnp.pi

    heading_error_int = state.heading_error_int + hd_error * cfg.dt
    heading_error_int = jnp.clip(heading_error_int, -cfg.heading_max_int_error, cfg.heading_max_int_error)

    delta  = kp *(hd_error) + kd*(hd_error - state.heading_error_prev)/cfg.dt + ki*heading_error_int

    #servo rate limits

    delta_change = delta - state.delta_prev
    delta_change = jnp.clip(delta_change, -cfg.delta_rate_max*cfg.dt, cfg.delta_rate_max*cfg.dt)
    delta = state.delta_prev + delta_change
    delta = jnp.clip(delta, -cfg.delta_max, cfg.delta_max)


    speed_error = ux - state.desired_ux

    speed_error = speed_error / cfg.max_ux


    speed_error_int = state.speed_error_int + speed_error * cfg.dt
    speed_error_int = jnp.clip(speed_error_int, -cfg.speed_max_int_error, cfg.speed_max_int_error)

    throttle = v_kp * speed_error + v_kd * (speed_error - state.velocity_error_prev)/cfg.dt + v_ki*speed_error_int


    # squash to throttle
    throttle = 0.5*(jnp.tanh(throttle) + 1.0)

    A,omega = pick_A_omega_from_S_linear_batched(throttle, cfg)



    A = A
    w = omega/2*jnp.pi
    alpha = A * (omega**2) * jnp.cos(omega*state.t)

    inp = make_input(state.t, alpha, delta)

    # ================= DYNAMICS =================
    x_next = dynamics_step(state.x, inp, state.params, cfg.dt)
    t_next = state.t + cfg.dt

    qh = x_next[:, 2]
    u  = x_next[:, 3]
    qh_dot = x_next[:, -1]
    head_x_prev, head_y_prev = head_position(state)
    # ================= TAIL =================
    tail_xpos_next, tail_ypos_next = update_tail_position(
        state.tail_xpos,
        state.tail_ypos,
        u,
        qh,
        cfg.dt
    )

    # ================= HEAD =================
    geom_state = state.replace(
        x=x_next,
        tail_xpos=tail_xpos_next,
        tail_ypos=tail_ypos_next,
        delta_prev=delta,
        head_x_prev=head_x_prev,
        head_y_prev=head_y_prev,
    )

    x_head, y_head = head_position(geom_state)

    # ================= VELOCITY =================
    vx, vy = world_velocity(
        x_head,
        y_head,
        head_x_prev,
        head_y_prev,
        cfg.dt,
    )

    ux, uy = body_velocity(vx, vy, qh)
    omega = qh_dot

    # ================= PATH =================
    ct_err, hd_err, path_heading, idx = compute_path_errors(
        state.paths,
        x_head,
        y_head,
        qh,
    )
    target, heading_des, new_idx, found = circle_lookahead(
        state.paths,
        x_head,
        y_head,
        state.path_idx,
        L=state.L
    )

    hd_err  = qh - heading_des
    hd_err = jnp.arctan2(jnp.sin(hd_err), jnp.cos(hd_err))
    # ================= EMA =================
    beta = cfg.beta

    x_avg = (1-beta)*state.head_x_avg + beta*x_head
    y_avg = (1-beta)*state.head_y_avg + beta*y_head
    heading_avg = (1-beta)*state.heading_avg + beta*qh

    ux_avg = (1-beta)*state.ux_avg + beta*ux
    uy_avg = (1-beta)*state.uy_avg + beta*uy
    omega_avg = (1-beta)*state.omega_avg + beta*omega

    # ================= RETURN CANDIDATE =================
    candidate_state = state.replace(
        x=x_next,
        tail_xpos=tail_xpos_next,
        tail_ypos=tail_ypos_next,
        head_x_prev=head_x_prev,
        head_y_prev=head_y_prev,



        head_x_avg=x_avg,
        head_y_avg=y_avg,
        head_x_avg_prev=state.head_x_avg,
        head_y_avg_prev=state.head_y_avg,

        heading_avg=heading_avg,
        prev_heading_avg=state.heading_avg,

        ux_avg=ux_avg,
        uy_avg=uy_avg,
        omega_avg=omega_avg,

        delta_prev=delta,
        alpha_prev=alpha,

        path_idx=new_idx,
        heading_desired=heading_des,
        desired_ux=state.desired_ux,

        kp=kp,
        kd=kd,
        ki=ki,
        v_kp=v_kp,
        v_kd=v_kd,
        v_ki=v_ki,
        heading_error_prev=hd_error,

        heading_error_int=heading_error_int,
        velocity_error_prev=speed_error,
        speed_error_int=speed_error_int,
        throttle_prev=throttle,

        L = state.L,

        t=t_next,
    )

    return candidate_state
@jax.jit
def step_env(state: EnvState, action, key, cfg: EnvConfig):



    candidate_state = step_core(state, action, cfg)

    done_next = compute_done(candidate_state, cfg)
    candidate_state = candidate_state.replace(done=done_next)

    reset_state = reset_env(key, state.x.shape[0], state.x.shape[1], cfg)

    def select(new, old):
        mask = done_next
        for _ in range(old.ndim - 1):
            mask = mask[:, None]
        return jnp.where(mask, new, old)

    new_state = jax.tree.map(select, reset_state, candidate_state)

    return new_state

@jax.jit
def eval_step_env(state: EnvState, action, key, cfg: EnvConfig):


    new_state = step_core(state, action, cfg)

    # No reset. No masking.
    return new_state
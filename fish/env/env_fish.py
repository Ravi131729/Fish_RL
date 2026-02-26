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

def make_input(t, alpha,delta):


    return jnp.stack(
        [delta,
         jnp.zeros_like(delta),
         jnp.zeros_like(delta),
         alpha],
        axis=1,
    )

@jax.jit
def step_core(state: EnvState, action, cfg: EnvConfig):

    kp_raw = action["kp"]   # [-1,1]
    kd_raw = action["kd"]   # [-1,1]
    L_raw = action["L"]     # [-1,1]

    kp_min, kp_max = 0.0, 5.0
    kd_min, kd_max = 0.0, 5.0
    L_min, L_max = 0.15, 0.5

    kp = kp_min + 0.5*(kp_raw + 1.0) * (kp_max - kp_min)
    kd = kd_min + 0.5*(kd_raw + 1.0) * (kd_max - kd_min)
    L = L_min + 0.5*(L_raw + 1.0) * (L_max - L_min)
    qh = state.x[:, 2]
    hd_error = qh - state.heading_desired
    hd_error = jnp.arctan2(jnp.sin(hd_error), jnp.cos(hd_error))
    delta  = kp *(hd_error) + kd*(hd_error - state.heading_error_prev)/cfg.dt

    delta_change = delta - state.delta_prev
    delta_change = jnp.clip(delta_change, -cfg.delta_rate_max*cfg.dt, cfg.delta_rate_max*cfg.dt)
    delta = state.delta_prev + delta_change
    delta = jnp.clip(delta, -cfg.delta_max, cfg.delta_max)

    A = state.A
    w = state.w
    alpha = A * ((2*jnp.pi*w)**2) * jnp.cos(2*jnp.pi*w*state.t)

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
        L=L
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

        kp=kp,
        kd=kd,
        heading_error_prev=hd_error,
        L = L,

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
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
from fish.utils.path_utils import compute_path_errors
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
def step_core(state: EnvState, delta_raw, cfg: EnvConfig):

    # ================= CONTROL =================
    delta_change = cfg.delta_rate_max * cfg.dt * delta_raw
    delta = jnp.clip(state.delta_prev + delta_change,
                     -cfg.delta_max, cfg.delta_max)

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
    )

    x_head, y_head = head_position(geom_state)

    # ================= VELOCITY =================
    vx, vy = world_velocity(
        x_head,
        y_head,
        state.head_x_prev,
        state.head_y_prev,
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

        head_x_prev=x_head,
        head_y_prev=y_head,

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

        path_idx=idx,
        heading_desired=path_heading,

        t=t_next,
    )

    return candidate_state
@jax.jit
def step_env(state: EnvState, action, key, cfg: EnvConfig):

    delta_raw = action["delta"]

    candidate_state = step_core(state, delta_raw, cfg)

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

    delta_raw = action["delta"]

    new_state = step_core(state, delta_raw, cfg)

    # No reset. No masking.
    return new_state
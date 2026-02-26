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
import jax
import jax.numpy as jnp

FSET = jnp.array([2.0,2.2,2.4,2.6,2.8, 3.0])


@jax.jit
def throttle_to_A_f(T, f_prev , cfg: EnvConfig):
    """
    T: scalar or (N,)
    f_prev: same shape as T
    returns: A, f
    """

    S = T / (4 * jnp.pi**2)  # S = A f^2

    # expand dims for broadcasting
    S_exp = S[..., None]          # (...,1)
    f_exp = FSET[None, ...]       # (1,5)

    A_all = S_exp / (f_exp**2)    # (...,5)

    valid = (A_all >= cfg.A_min) & (A_all <= cfg.A_max)

    # distance to previous frequency
    freq_dist = jnp.abs(f_exp - f_prev[..., None])

    # mask invalid
    big = 1e6
    cost = jnp.where(valid, freq_dist, big)

    # pick best valid
    idx = jnp.argmin(cost, axis=-1)

    A_pick = jnp.take_along_axis(A_all, idx[..., None], axis=-1)[...,0]
    f_pick = FSET[idx]

    # -------- fallback if NO valid solution ----------
    any_valid = jnp.any(valid, axis=-1)

    # clamp A and choose closest throttle match
    A_clamped = jnp.clip(A_all, cfg.A_min, cfg.A_max)
    S_hat = A_clamped * (f_exp**2)
    err = jnp.abs(S_hat - S_exp)

    idx_fb = jnp.argmin(err, axis=-1)
    A_fb = jnp.take_along_axis(A_clamped, idx_fb[...,None], axis=-1)[...,0]
    f_fb = FSET[idx_fb]

    A_final = jnp.where(any_valid, A_pick, A_fb)
    f_final = jnp.where(any_valid, f_pick, f_fb)

    return A_final, f_final

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
    delta_raw = action["delta"]
    throttle_raw = action["throttle"]

    # ================= CONTROL =================
    delta_change = cfg.delta_rate_max * cfg.dt * delta_raw
    delta = jnp.clip(state.delta_prev + delta_change,
                     -cfg.delta_max, cfg.delta_max)
    throttle_change = 1000 * cfg.dt * throttle_raw

    throttle_min =946
    throttle_max = cfg.alpha_max
    throttle = jnp.clip(state.throttle_prev + throttle_change,
                        throttle_min, throttle_max)
    A , f = throttle_to_A_f(throttle, state.w, cfg)

    A = A
    w = f
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
        throttle_prev=throttle,
        A=A,
        w=w,

        path_idx=idx,
        heading_desired=path_heading,

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
import jax.numpy as jnp
from dataclasses import dataclass
from fish.env.physics_params import PhysicsParams,get_constants
from fish.env.integrator import rk4_step
import jax
import chex
import jax.numpy as jnp
import time
from fish.utils.path_utils import  wrap_to_pi, make_input , compute_path_errors,sample_paths_batch,get_lookahead_body, circle_lookahead_intersection

@chex.dataclass
class EnvConfig:
    # integration
    dt: float
    max_steps: int

    # direct change in control bounds
    delta_max: float
    delta_rate_max: float
    alpha_rate_max: float
    alpha_max: float
    # EMA smoothing factor for head position
    beta: float
    max_heading_angle: float
    min_heading_angle: float
    max_ux: float
    min_ux: float

    omega_rotor_max: float

    A_min : float
    A_max : float
    w_min : float
    w_max : float

    max_position: float


@chex.dataclass
class EnvState:
    x: jnp.ndarray
    tail_xpos: jnp.ndarray
    tail_ypos: jnp.ndarray

    head_x_prev: jnp.ndarray
    head_y_prev: jnp.ndarray

    head_x_avg: jnp.ndarray
    head_y_avg: jnp.ndarray

    # previous avg for velocity(world_frame)
    head_x_avg_prev: jnp.ndarray
    head_y_avg_prev: jnp.ndarray

    heading_avg: jnp.ndarray
    prev_heading_avg: jnp.ndarray

    #body frame
    ux_avg: jnp.ndarray
    uy_avg: jnp.ndarray
    omega_avg: jnp.ndarray


    heading_desired: jnp.ndarray

    A:  jnp.ndarray
    w:  jnp.ndarray

    delta_prev: jnp.ndarray  # servo angle
    alpha_prev: jnp.ndarray
    # omega_rotor_prev: jnp.ndarray
    t: jnp.ndarray

    params: PhysicsParams
    done: jnp.ndarray

    paths: jnp.ndarray
    path_idx: jnp.ndarray



def add_obs_noise(key, obs, sigma=0.01):
    noise = sigma * jax.random.normal(key, obs.shape)
    return obs + noise

def build_obs(state: EnvState, cfg: EnvConfig,key):

    dt = cfg.dt
    xpos, ypos = head_position(state)

    vx = (xpos - state.head_x_prev) / dt
    vy = (ypos - state.head_y_prev) / dt
    qh = state.x[:, 2]
    qdh = state.x[:, -1]

    #body ux, body uy
    c = jnp.cos(qh)
    s = jnp.sin(qh)

    u_x =  vx * c + vy * s     # forward velocity (body)
    u_y = -vx * s + vy * c     # lateral velocity (body)

    ux_avg = state.ux_avg
    uy_avg = state.uy_avg
    omega_avg = state.omega_avg



    # path errors
    vx_avg = (state.head_x_avg - state.head_x_avg_prev) / dt
    vy_avg = (state.head_y_avg - state.head_y_avg_prev) / dt
    heading_avg = state.heading_avg

    qh = state.x[:, 2]
    u = state.x[:, 3]

    ct_err, hd_err, path_heading, idx = compute_path_errors(
        state.paths,
        xpos,
        ypos,
        qh,
    )
    b1, b2, b3 = get_lookahead_body(state.paths, idx, xpos, ypos, qh)
    # target_pt, heading_desired, found = circle_lookahead_intersection(state.paths,xpos, ypos, qh, L=0.2)
    # hd_err = wrap_to_pi(qh - heading_desired)




    obs_clean = jnp.concatenate([
            u_x[:, None],
            u_y[:, None],
            qh[:, None],

            ct_err[:, None],
            hd_err[:, None],
            state.delta_prev[:, None],

    ], axis=1)
    key, subkey = jax.random.split(key)
    obs_noisy = add_obs_noise(subkey, obs_clean, sigma=0.01)

    return obs_noisy

def sample_physics_params(key, N):
    keys = jax.random.split(key, 5)

    return PhysicsParams(
        b1=jnp.full((N,), 0.075),
        bs=jnp.full((N,), 0.035),
        l11=jnp.full((N,), 0.048),
        l21=jnp.full((N,), 0.048),
        ls1=jnp.full((N,), 0.015),
        c1=jnp.full((N,), 0.03),
        #5% randomization around nominal values
        added_mass_scale=jax.random.uniform(keys[0], (N,), minval=0.95, maxval=1.05),
        inertia_scale=jax.random.uniform(keys[1], (N,), minval=0.95, maxval=1.05),
        head_damping_scale=jax.random.uniform(keys[2], (N,), minval=0.95, maxval=1.05),
        link_damping_scale=jax.random.uniform(keys[3], (N,), minval=0.95, maxval=1.05),
        stiffness_scale=jax.random.uniform(keys[4], (N,), minval=0.95, maxval=1.05),
        #10% randomization around nominal values
        # added_mass_scale=jax.random.uniform(keys[0], (N,), minval=0.90, maxval=1.10),
        # inertia_scale=jax.random.uniform(keys[1], (N,), minval=0.90, maxval=1.10),
        # head_damping_scale=jax.random.uniform(keys[2], (N,), minval=0.90, maxval=1.10),
        # link_damping_scale=jax.random.uniform(keys[3], (N,), minval=0.90, maxval=1.10),
        # stiffness_scale=jax.random.uniform(keys[4], (N,), minval=0.90, maxval=1.10),


    )
def head_position(state: EnvState):
    q1  = state.x[:, 0]
    q2  = state.x[:, 1]
    qh  = state.x[:, 2]

    l1  = state.params.l11
    l2  = state.params.l21
    ls1 = state.params.ls1
    b   = state.params.b1
    delta = state.delta_prev
    # world-frame head position
    xpos = (
        state.tail_xpos
        + l1  * jnp.cos(q1)
        + l2  * jnp.cos(q2)
        + ls1 * jnp.cos(qh - delta)
        + b   * jnp.cos(qh)
    )

    ypos = (
        state.tail_ypos
        + l1  * jnp.sin(q1)
        + l2  * jnp.sin(q2)
        + ls1 * jnp.sin(qh - delta)
        + b   * jnp.sin(qh)
    )

    return xpos, ypos


def compute_reward(state, state_next,action, cfg):

    delta_raw = action["delta"]
    qh = state.x[:, 2]
    u = state.x[:, 3]
    xpos, ypos = head_position(state)

    vx = (xpos - state.head_x_prev) / cfg.dt
    vy = (ypos - state.head_y_prev) / cfg.dt
    qh = state.x[:, 2]

    dt = cfg.dt
    # path errors
    vx_avg = (state.head_x_avg - state.head_x_avg_prev) / dt
    vy_avg = (state.head_y_avg - state.head_y_avg_prev) / dt
    heading_avg = state.heading_avg

    #body ux, body uy
    c = jnp.cos(qh)
    s = jnp.sin(qh)

    u_x =  vx * c + vy * s
    u_y = -vx * s + vy * c

    qdh = state.x[:, -1]
    heading_avg = state.heading_avg
    ux_avg = state.ux_avg
    uy_avg = state.uy_avg
    omega_avg = state.omega_avg

    ct_err, hd_err, path_heading, idx = compute_path_errors(
        state.paths,
        xpos,
        ypos,
        qh,
    )
    N_path = state.paths.shape[1]  # 256

    raw = (idx - state.path_idx + N_path) % N_path    # in [0, N_path-1]
    progress = jnp.clip(raw, 0.0, 5.0)

    # target_pt, heading_desired, found = circle_lookahead_intersection(state.paths, xpos, ypos, qh, L=0.2)
    # hd_err = wrap_to_pi(qh - heading_desired)
    hd = hd_err / jnp.pi

    delta_change = state_next.delta_prev - state.delta_prev

    reward = (
        # + 0.5 * progress
        - 2.0 * ct_err**2         # stay near path
        - 0.5 * hd**2             #  heading
        - 0.01 * delta_change**2     # smooth control
    )


    info = {
    "ux_avg": ux_avg,
    "uy_avg": uy_avg,
    "heading_avg": heading_avg,
    "omega_avg": progress,
    "qh": qh,
    "u": u,
    "qdh": qdh,
    "u_x": u_x,
    "u_y": u_y,
    "heading_error": hd_err,
    "cross_track_error": ct_err,


}

    return jnp.clip(reward, -3.0, 3.0),     info

def reset_env(key, N, nx, cfg):

    k1,k2,k3,k4,k5,k6,k7,k8 = jax.random.split(key,8)
    paths = sample_paths_batch(k8, N)   # (N,256,2)
    p0 = paths[:, 0, :]
    p1 = paths[:, 1, :]
    heading_desired = jnp.arctan2(p1[:,1]-p0[:,1], p1[:,0]-p0[:,0])

    init_heading = heading_desired + jax.random.uniform(k1, (N,), minval=-1, maxval=1)
    # --------------------------------------------------
    # initial state
    # --------------------------------------------------
    x0 = jnp.zeros((N,nx))  # your internal fish state
    x0 = x0.at[:, 0].set(init_heading)
    x0 = x0.at[:, 1].set(init_heading)
    x0 = x0.at[:, 2].set(init_heading)

    delta0 = jnp.zeros((N,))
    alpha0 = jnp.zeros((N,))
    t0 = jnp.zeros((N,))
    omega_rotor0 = jnp.zeros((N,))

    kA, kW = jax.random.split(k7)
    A = jax.random.uniform(kA, (N,), minval=cfg.A_min, maxval=cfg.A_max)
    w = jax.random.uniform(kW, (N,), minval=cfg.w_min, maxval=cfg.w_max)

    # Sample initial values for tail_xpos, tail_ypos, head_x_prev, head_y_prev, delta, A, w within their respective limits
    start = paths[:,0,:]

    tail_xpos = start[:,0]
    tail_ypos = start[:,1]

    head_x_prev = tail_xpos
    head_y_prev = tail_ypos



    state = EnvState(
        x=x0,
        tail_xpos=tail_xpos,
        tail_ypos=tail_ypos,

        head_x_prev=head_x_prev,
        head_y_prev=head_y_prev,


        head_x_avg=head_x_prev,
        head_y_avg=head_y_prev,
        head_x_avg_prev=head_x_prev,
        head_y_avg_prev=head_y_prev,

        heading_avg=x0[:, 2],  # initial heading from state
        prev_heading_avg=x0[:, 2],

        delta_prev=delta0,
        alpha_prev=alpha0,
        # omega_rotor_prev=omega_rotor0,
        #body frame velocity
        ux_avg=jnp.zeros((N,)),
        uy_avg=jnp.zeros((N,)),
        omega_avg=jnp.zeros((N,)),
        # ux_desired=ux_desired,
        heading_desired=heading_desired,
        paths=paths,
        path_idx=jnp.zeros((N,), dtype=jnp.int32),

        t=t0,

        A=A,
        w=w,

        params=sample_physics_params(k3, N),
        done=jnp.zeros((N,), dtype=bool),
          # increment update step for curriculum

    )

    # --------------------------------------------------
    # initialize prev head pos
    # --------------------------------------------------
    xpos, ypos = head_position(state)

    state = state.replace(
        head_x_prev=xpos,
        head_y_prev=ypos,
        head_x_avg=xpos,
        head_y_avg=ypos,
        head_x_avg_prev=xpos,
        head_y_avg_prev=ypos,
    )

    return state

def env_step_single(x, inp, params, dt):
    const = get_constants(params)
    return rk4_step(x, inp, const, dt)

env_step_vmap = jax.jit(
    jax.vmap(env_step_single, in_axes=(0, 0, 0, None))
)
def compute_done(state: EnvState, cfg: EnvConfig):
    # --- state-based termination ---
    psi = state.x[:, 2]
    # heading_bounds = jnp.abs(state.heading_avg) > jnp.pi / 6
    heading_bounds = jnp.abs(psi) > 6 * jnp.pi
    out_of_bounds = jnp.linalg.norm(state.x[:, :2], axis=1) >100

    nan_state = jnp.any(jnp.isnan(state.x), axis=1)

    # --- time limit termination ---
    time_limit = state.t >= cfg.max_steps * cfg.dt
    # big_error = jnp.abs(e_ct_now) >2.0
    # --- combine ALL termination conditions ---
    done = (
        out_of_bounds
        | nan_state
        | time_limit
        | heading_bounds
        # | big_error
    )

    return done
@jax.jit
def step_env(state: EnvState,action, key, cfg: EnvConfig):
    delta_raw = action["delta"]

    A = state.A
    w = state.w
    alpha = A*((2*jnp.pi*w)**2)*jnp.cos(2*jnp.pi*w*state.t)

    # delta = jnp.zeros_like(alpha) # for testing only, remove later
    delta_change = cfg.delta_rate_max*cfg.dt * delta_raw
    delta = state.delta_prev + delta_change
    delta = jnp.clip(delta, -cfg.delta_max, cfg.delta_max)

    # --- build control ---
    inp = make_input(state.t, alpha,delta)
    qh  = state.x[:, 2]
    u   = state.x[:, 3]
    qh_dot = state.x[:, -1]

    xd = u * jnp.cos(qh)
    yd = u * jnp.sin(qh)
    head_x_prev, head_y_prev = head_position(state)
    # --- dynamics ---
    x_next = env_step_vmap(state.x, inp, state.params, cfg.dt)
    t_next = state.t + cfg.dt
    tail_xpos_next = state.tail_xpos + xd*cfg.dt
    tail_ypos_next = state. tail_ypos + yd*cfg.dt

    # --- termination (based on NEXT state) ---
    temp_state = EnvState(
        x=x_next,
        tail_xpos = tail_xpos_next,
        tail_ypos = tail_ypos_next,
        head_x_prev = head_x_prev,
        head_y_prev = head_y_prev,
        head_x_avg = state.head_x_avg,
        head_y_avg = state.head_y_avg,
        head_x_avg_prev = state.head_x_avg,
        head_y_avg_prev = state.head_y_avg,
        heading_avg = state.heading_avg,
        prev_heading_avg = state.heading_avg,
        delta_prev =delta,
        alpha_prev =alpha,

        ux_avg = state.ux_avg,
        uy_avg = state.uy_avg,
        omega_avg = state.omega_avg,

        heading_desired = state.heading_desired,


        t=t_next,
        A=state.A,
        w=state.w,
        paths=state.paths,
        path_idx=state.path_idx,

        params=state.params,
        done=state.done,

    )
    x_head, y_head = head_position(temp_state)

    vx = (x_head - head_x_prev) / cfg.dt
    vy = (y_head - head_y_prev) / cfg.dt
    qh = x_next[:, 2]
    u = x_next[:, 3]
    c = jnp.cos(qh)
    s = jnp.sin(qh)
    #body frame velocity
    ux =  vx * c + vy * s     # forward velocity (body)
    uy = -vx * s + vy * c     # lateral velocity (body)
    omega = qh_dot

    ct_err, hd_err, path_heading, idx = compute_path_errors(
        state.paths,
        x_head,
        y_head,
        qh,
    )
    # EMA position
    x_avg = (1.0 - cfg.beta) * state.head_x_avg + cfg.beta * x_head
    y_avg = (1.0 - cfg.beta) * state.head_y_avg + cfg.beta * y_head
    heading_avg = (1.0 - cfg.beta) * state.heading_avg + cfg.beta * temp_state.x[:, 2]
    # EMA velocity in body frame
    ux_avg = (1.0 - cfg.beta) * state.ux_avg + cfg.beta * ux
    uy_avg = (1.0 - cfg.beta) * state.uy_avg + cfg.beta * uy
    omega_avg = (1.0 - cfg.beta) * state.omega_avg + cfg.beta * omega
    done_next = compute_done(temp_state, cfg)
    temp_state = temp_state.replace(head_x_avg=x_avg, head_y_avg=y_avg, heading_avg=heading_avg, ux_avg=ux_avg, uy_avg=uy_avg, omega_avg=omega_avg,path_idx=idx, heading_desired=path_heading)


    reset_state = reset_env(key, state.x.shape[0], state.x.shape[1], cfg)
    candidate_state= EnvState(
                            x=x_next,
                            tail_xpos = tail_xpos_next,
                            tail_ypos = tail_ypos_next,
                            head_x_prev = head_x_prev,
                            head_y_prev = head_y_prev,
                            head_x_avg = x_avg,
                            head_y_avg = y_avg,
                            head_x_avg_prev = state.head_x_avg,
                            head_y_avg_prev = state.head_y_avg,
                            heading_avg = heading_avg,
                            prev_heading_avg = state.heading_avg,
                            delta_prev =delta,
                            alpha_prev =alpha,

                            ux_avg = ux_avg,
                            uy_avg = uy_avg,
                            omega_avg = omega_avg,

                            heading_desired = path_heading,
                            paths=state.paths,
                            path_idx=idx,

                            A=state.A,
                            w=state.w,
                            t=t_next,
                            params=state.params,
                            done=done_next,

                        )

    # --- selective per-env reset ---
    def select_reset(new, old, done_next):
        mask = done_next
        for _ in range(old.ndim - 1):
            mask = mask[:, None]
        return jnp.where(mask, new, old)

    new_state = jax.tree.map(lambda new, old: select_reset(new, old, done_next),
                            reset_state, candidate_state)


    return new_state
@jax.jit
def eval_step_env(state: EnvState, delta_raw, key, cfg: EnvConfig):






    delta_change = cfg.delta_rate_max*cfg.dt * delta_raw
    delta = state.delta_prev + delta_change
    delta = jnp.clip(delta, -cfg.delta_max, cfg.delta_max)
    A = state.A
    w = state.w
    alpha = A*((2*jnp.pi*w)**2)*jnp.cos(2*jnp.pi*w*state.t)  # open-loop oscillation for evaluation
    # --- build control ---
    inp = make_input(state.t, alpha,delta)
    qh  = state.x[:, 2]
    u   = state.x[:, 3]
    qh_dot = state.x[:, -1]

    xd = u * jnp.cos(qh)
    yd = u * jnp.sin(qh)
    head_x_prev, head_y_prev = head_position(state)
    # --- dynamics ---
    x_next = env_step_vmap(state.x, inp, state.params, cfg.dt)
    t_next = state.t + cfg.dt
    tail_xpos_next = state.tail_xpos + xd*cfg.dt
    tail_ypos_next = state. tail_ypos + yd*cfg.dt


    # --- termination (based on NEXT state) ---
    temp_state = EnvState(
        x=x_next,
        tail_xpos = tail_xpos_next,
        tail_ypos = tail_ypos_next,
        head_x_prev = head_x_prev,
        head_y_prev = head_y_prev,
        head_x_avg = state.head_x_avg,
        head_y_avg = state.head_y_avg,
        head_x_avg_prev = state.head_x_avg,
        head_y_avg_prev = state.head_y_avg,
        heading_avg = state.heading_avg,
        prev_heading_avg = state.heading_avg,
        delta_prev =delta,
        alpha_prev =alpha,
        # omega_rotor_prev = omega_rotor,
        ux_avg = state.ux_avg,
        uy_avg = state.uy_avg,
        omega_avg = state.omega_avg,
        # ux_desired = ux_desired,
        heading_desired = state.heading_desired,
        paths=state.paths,
        path_idx=state.path_idx,

        t=t_next,
        # v_timer=v_timer,
        A=state.A,
        w=state.w,

        params=state.params,
        done=state.done,

    )
    x_head, y_head = head_position(temp_state)
    ct_err, hd_err, path_heading, idx = compute_path_errors(
        state.paths,
        x_head,
        y_head,
        qh,
    )

    vx = (x_head - head_x_prev) / cfg.dt
    vy = (y_head - head_y_prev) / cfg.dt
    qh = x_next[:, 2]
    u = x_next[:, 3]
    c = jnp.cos(qh)
    s = jnp.sin(qh)
    #body frame velocity
    ux =  vx * c + vy * s     # forward velocity (body)
    uy = -vx * s + vy * c     # lateral velocity (body)
    omega = qh_dot

    # EMA position
    x_avg = (1.0 - cfg.beta) * state.head_x_avg + cfg.beta * x_head
    y_avg = (1.0 - cfg.beta) * state.head_y_avg + cfg.beta * y_head
    heading_avg = (1.0 - cfg.beta) * state.heading_avg + cfg.beta * temp_state.x[:, 2]
    # EMA velocity in body frame
    ux_avg = (1.0 - cfg.beta) * state.ux_avg + cfg.beta * ux
    uy_avg = (1.0 - cfg.beta) * state.uy_avg + cfg.beta * uy
    omega_avg = (1.0 - cfg.beta) * state.omega_avg + cfg.beta * omega
    # done_next = compute_done(temp_state, cfg)
    temp_state = temp_state.replace(head_x_avg=x_avg, head_y_avg=y_avg, heading_avg=heading_avg, ux_avg=ux_avg, uy_avg=uy_avg, omega_avg=omega_avg,path_idx=idx, heading_desired=path_heading)


    return temp_state



import jax
import jax.numpy as jnp
from fish.env.types import EnvState, EnvConfig
from fish.env.physics_params import PhysicsParams
from fish.utils.path_utils import sample_paths_batch
from fish.env.kinematics import head_position


# ==========================================================
# Physics parameter sampling (domain randomization)
# ==========================================================
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

# ==========================================================
# Environment reset
# ==========================================================
def reset_env(key, N, nx, cfg: EnvConfig):

    k1,k2,k3,k4,k5 = jax.random.split(key,5)

    # =====================================================
    # sample path
    # =====================================================
    paths = sample_paths_batch(k4, N)   # (N,256,2)

    p0 = paths[:,0,:]
    p1 = paths[:,1,:]

    heading_desired = jnp.arctan2(
        p1[:,1]-p0[:,1],
        p1[:,0]-p0[:,0]
    )

    # =====================================================
    # initial heading
    # =====================================================
    init_heading = heading_desired + jax.random.uniform(
        k2, (N,), minval=-1.0, maxval=1.0
    )

    # =====================================================
    # internal state x
    # =====================================================
    x0 = jnp.zeros((N,nx))
    x0 = x0.at[:,0].set(init_heading)
    x0 = x0.at[:,1].set(init_heading)
    x0 = x0.at[:,2].set(init_heading)

    # =====================================================
    # propulsion parameters
    # =====================================================
    kA, kW = jax.random.split(k3)

    A = jax.random.uniform(kA, (N,), minval=cfg.A_min, maxval=cfg.A_max)
    w = jax.random.uniform(kW, (N,), minval=cfg.w_min, maxval=cfg.w_max)

    keyp1, keyp2, keyp3, keyp4, keyp5 , keyp6,keyp7 = jax.random.split(k5, 7)

    kp = jax.random.uniform(keyp1, (N,), minval=0.0, maxval=cfg.max_kp)
    kd = jax.random.uniform(keyp2, (N,), minval=0.0, maxval=cfg.max_kd)
    ki = jax.random.uniform(keyp3, (N,), minval=0.0, maxval=cfg.max_ki)

    v_kp = jax.random.uniform(keyp4, (N,), minval=0.0, maxval=cfg.max_v_kp)
    v_kd = jax.random.uniform(keyp5, (N,), minval=0.0, maxval=cfg.max_v_kd)
    v_ki = jax.random.uniform(keyp6, (N,), minval=0.0, maxval=cfg.max_v_ki)

    L= 0.25*jnp.ones((N,))

    desired_ux = jax.random.uniform(keyp6, (N,), minval=cfg.min_ux, maxval=cfg.max_ux)



    # =====================================================
    # start position = path start
    # =====================================================
    start = paths[:,0,:]
    tail_xpos = start[:,0]
    tail_ypos = start[:,1]

    # temporary state for head computation
    dummy_state = EnvState(
        x=x0,
        tail_xpos=tail_xpos,
        tail_ypos=tail_ypos,

        head_x_prev=tail_xpos,
        head_y_prev=tail_ypos,

        head_x_avg=tail_xpos,
        head_y_avg=tail_ypos,
        head_x_avg_prev=tail_xpos,
        head_y_avg_prev=tail_ypos,

        heading_avg=x0[:,2],
        prev_heading_avg=x0[:,2],

        ux_avg=jnp.zeros((N,)),
        uy_avg=jnp.zeros((N,)),
        omega_avg=jnp.zeros((N,)),

        delta_prev=jnp.zeros((N,)),
        alpha_prev=jnp.zeros((N,)),

        A=A,
        w=w,

        paths=paths,
        path_idx=jnp.zeros((N,),dtype=jnp.int32),
        heading_desired=heading_desired,
        desired_ux=desired_ux,

        kp=kp,
        kd=kd,
        ki=ki,
        v_kp=v_kp,
        v_kd=v_kd,
        v_ki=v_ki,
        L=L,

        throttle_prev=jnp.zeros((N,)),

        heading_error_prev=jnp.zeros((N,)),
        velocity_error_prev=jnp.zeros((N,)),

        heading_error_int=jnp.zeros((N,)),
        speed_error_int=jnp.zeros((N,)),



        t=jnp.zeros((N,)),
        params=sample_physics_params(k4,N),
        done=jnp.zeros((N,),dtype=bool),
    )

    # =====================================================
    # compute correct head position
    # =====================================================
    head_x, head_y = head_position(dummy_state)

    # =====================================================
    # final state
    # =====================================================
    state = dummy_state.replace(
        head_x_prev=head_x,
        head_y_prev=head_y,
        head_x_avg=head_x,
        head_y_avg=head_y,
        head_x_avg_prev=head_x,
        head_y_avg_prev=head_y,
    )

    return state

def compute_done(state: EnvState, cfg: EnvConfig):
    # --- state-based termination ---
    psi = state.x[:, 2]
    # heading_bounds = jnp.abs(state.heading_avg) > jnp.pi / 6
    heading_bounds = jnp.abs(psi) > 6 * jnp.pi
    out_of_bounds = jnp.linalg.norm(state.x[:, :3], axis=1) >1.5

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

    )

    return done
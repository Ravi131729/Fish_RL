import chex
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict
from fish.env.physics_params import PhysicsParams


# ================================
# Configuration
# ================================

@chex.dataclass
class EnvConfig:
    # integration
    dt: float
    max_steps: int

    # control limits
    delta_max: float
    delta_rate_max: float
    alpha_rate_max: float
    alpha_max: float

    # smoothing
    beta: float

    # heading bounds
    max_heading_angle: float
    min_heading_angle: float

    # velocity limits
    max_ux: float
    min_ux: float

    # oscillation params (open loop tail)
    omega_rotor_max: float
    A_min: float
    A_max: float
    w_min: float
    w_max: float

    # termination
    max_position: float


# ================================
# Environment State
# ================================

@chex.dataclass
class EnvState:
    # ------------------------
    # Internal fish state
    # ------------------------
    x: jnp.ndarray                 # (N, nx)
    tail_xpos: jnp.ndarray         # (N,)
    tail_ypos: jnp.ndarray         # (N,)

    # ------------------------
    # Previous head position
    # ------------------------
    head_x_prev: jnp.ndarray
    head_y_prev: jnp.ndarray

    # ------------------------
    # EMA filtered states
    # ------------------------
    head_x_avg: jnp.ndarray
    head_y_avg: jnp.ndarray
    head_x_avg_prev: jnp.ndarray
    head_y_avg_prev: jnp.ndarray

    heading_avg: jnp.ndarray
    prev_heading_avg: jnp.ndarray

    ux_avg: jnp.ndarray
    uy_avg: jnp.ndarray
    omega_avg: jnp.ndarray

    # ------------------------
    # Control memory
    # ------------------------
    delta_prev: jnp.ndarray
    alpha_prev: jnp.ndarray

    # ------------------------
    # Episode constants
    # ------------------------
    A: jnp.ndarray
    w: jnp.ndarray

    # ------------------------
    # Task state
    # ------------------------
    paths: jnp.ndarray             # (N, N_path, 2)
    path_idx: jnp.ndarray
    heading_desired: jnp.ndarray

    # ------------------------
    # Misc
    # ------------------------
    t: jnp.ndarray
    params: PhysicsParams
    done: jnp.ndarray


# ================================
# Action type
# ================================

Action = Dict[str, jnp.ndarray]
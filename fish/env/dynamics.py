import jax
import jax.numpy as jnp
from fish.env.physics_params import get_constants
from fish.env.integrator import rk4_step


# ==========================================================
# Single environment physics step
# ==========================================================
def step_single(x, inp, params, dt):
    const = get_constants(params)
    return rk4_step(x, inp, const, dt)


# ==========================================================
# Vectorized step for N environments
# ==========================================================
step_vmap = jax.jit(
    jax.vmap(step_single, in_axes=(0, 0, 0, None))
)



def step(x, inp, params, dt):
    """
    Main dynamics step used by environment.
    x: (N,nx)
    inp: (N,nu)
    params: PhysicsParams (N,)
    dt: float
    """
    return step_vmap(x, inp, params, dt)



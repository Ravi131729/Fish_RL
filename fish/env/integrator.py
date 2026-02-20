import jax.numpy as jnp
from jax.numpy.linalg import solve
import jax
from fish.dynamics.CS_4link_dynamics import mass_matrix, coriolis_vector, gravity_vector
from fish.env.physics_params import get_constants



def dynamics(states, inputs, const_vals):
    """
    states = (u, qd1, q1, qd2, q2, qdh, qh)
    inputs = (alpha, dalpha, ddalpha, ddphi)
    """
    args = (*states, *inputs, *const_vals)

    M = mass_matrix(*args)
    C = coriolis_vector(*args)
    G = gravity_vector(*args)
    return solve(M, -C - G).flatten()
def get_ordered_states(x):
    return jnp.array([x[3],x[4],x[0],x[5],x[1],x[6],x[2]])
def f(x, inp, const_vals):
    """
    x : state vector [q1, q2, qh, u, qd1,  qd2, qdh]
    u : input vector [alpha, dalpha, ddalpha, ddphi]
    """
    q_dot = jnp.array([x[4],x[5],x[6]])
    states = get_ordered_states(x)
    return jnp.concatenate([q_dot,
                     dynamics(states, inp, const_vals)])

def rk4_step(x, inp, const_vals, dt):
    k1 = f(x, inp, const_vals)
    k2 = f(x + 0.5 * dt * k1, inp, const_vals)
    k3 = f(x + 0.5 * dt * k2, inp, const_vals)
    k4 = f(x + dt * k3, inp, const_vals)

    return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)


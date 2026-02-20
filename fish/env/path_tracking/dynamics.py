"""
Fish Robot Dynamics (NumPy Version)

Contains dynamics equations and integration methods for the fish robot.
"""

import numpy as np
from numpy.linalg import solve

# Import from parent directory
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from numpy_matrices import mass_matrix, coriolis_vector, gravity_vector
from numpy_consts import get_constants


def dynamics(states, inputs, const_vals):
    """
    Compute accelerations from the dynamics equations.

    Args:
        states: (u, qd1, q1, qd2, q2, qdh, qh) - velocities and positions
        inputs: (alpha, dalpha, ddalpha, ddphi) - steering and propulsion
        const_vals: Physical constants

    Returns:
        Accelerations [du, dqd1, dqd2, dqdh]
    """
    args = (*states, *inputs, *const_vals)
    M = mass_matrix(*args)
    C = coriolis_vector(*args)
    G = gravity_vector(*args)
    return solve(M, -C - G).flatten()


def get_ordered_states(x):
    """
    Reorder state vector for dynamics function.

    Args:
        x: State [q1, q2, qh, u, qd1, qd2, qdh]

    Returns:
        Reordered state [u, qd1, q1, qd2, q2, qdh, qh]
    """
    return np.array([x[3], x[4], x[0], x[5], x[1], x[6], x[2]])


def f(x, inp, const_vals):
    """
    State derivative function for integration.

    Args:
        x: State vector [q1, q2, qh, u, qd1, qd2, qdh]
        inp: Input vector [delta, ddelta, dddelta, ddphi]
        const_vals: Physical constants

    Returns:
        State derivative dx/dt
    """
    q_dot = np.array([x[4], x[5], x[6]])
    states = get_ordered_states(x)
    return np.concatenate([q_dot, dynamics(states, inp, const_vals)])


def rk4_step(x, inp, const_vals, dt):
    """
    Runge-Kutta 4th order integration step.

    Args:
        x: Current state
        inp: Input vector
        const_vals: Physical constants
        dt: Time step

    Returns:
        Next state
    """
    k1 = f(x, inp, const_vals)
    k2 = f(x + 0.5 * dt * k1, inp, const_vals)
    k3 = f(x + 0.5 * dt * k2, inp, const_vals)
    k4 = f(x + dt * k3, inp, const_vals)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def compute_head_position(x, tail_xpos, tail_ypos, delta=0.0):
    """
    Compute the head position of the fish given state and tail position.

    Args:
        x: State vector [q1, q2, qh, u, qd1, qd2, qdh]
        tail_xpos: Tail x position
        tail_ypos: Tail y position
        delta: Steering angle (optional)

    Returns:
        (head_x, head_y): Head position
    """
    q1, q2, qh = x[0], x[1], x[2]

    # Link lengths (matching fish geometry)
    l1 = l2 = 0.048  # Tail segment lengths
    ls1 = 0.015      # Steering segment length
    b = 0.075 * 1.5  # Head length

    # Forward kinematics: sum of all link contributions
    xc = tail_xpos + l1 * np.cos(q1) + l2 * np.cos(q2) + ls1 * np.cos(qh - delta) + b * np.cos(qh)
    yc = tail_ypos + l1 * np.sin(q1) + l2 * np.sin(q2) + ls1 * np.sin(qh - delta) + b * np.sin(qh)

    return xc, yc


def wrap_to_pi(angle):
    """
    Wrap angle to [-pi, pi].

    Args:
        angle: Angle in radians

    Returns:
        Wrapped angle in [-pi, pi]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

import jax.numpy as jnp
from fish.env.types import EnvState


# ==========================================================
# Head Position (World Frame)
# ==========================================================

def head_position(state: EnvState):
    """
    Computes world-frame head position from internal joint state.
    Vectorized over N environments.
    """

    q1  = state.x[:, 0]
    q2  = state.x[:, 1]
    qh  = state.x[:, 2]

    l1  = state.params.l11
    l2  = state.params.l21
    ls1 = state.params.ls1
    b   = state.params.b1

    delta = state.delta_prev

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

# ==========================================================
# World velocity from position difference
# ==========================================================

def world_velocity(x_head, y_head, x_prev, y_prev, dt):
    vx = (x_head - x_prev) / dt
    vy = (y_head - y_prev) / dt
    return vx, vy

# ==========================================================
# Body-frame velocity transform
# ==========================================================

def body_velocity(vx, vy, heading):
    c = jnp.cos(heading)
    s = jnp.sin(heading)

    u_x =  vx * c + vy * s
    u_y = -vx * s + vy * c

    return u_x, u_y

# ==========================================================
# Tail position update (simple rigid drift)
# ==========================================================

def update_tail_position(tail_x, tail_y, u, heading, dt):
    xd = u * jnp.cos(heading)
    yd = u * jnp.sin(heading)

    tail_x_next = tail_x + xd * dt
    tail_y_next = tail_y + yd * dt

    return tail_x_next, tail_y_next
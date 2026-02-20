import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

POOL_X = 3.0
POOL_Y = 2.0
N_PATH = 20  # <<< SAME FOR ALL


# ============================================================
# CIRCLE
# ============================================================
def make_circle_pool(key):
    k1, k2, k3 = jax.random.split(key, 3)

    r = jax.random.uniform(k1, (), minval=0.5, maxval=0.9)

    cx = jax.random.uniform(k2, (), minval=r, maxval=POOL_X-r)
    cy = jax.random.uniform(k3, (), minval=r, maxval=POOL_Y-r)

    theta = jnp.linspace(0, 2*jnp.pi, N_PATH)

    x = cx + r*jnp.cos(theta)
    y = cy + r*jnp.sin(theta)

    return jnp.stack([x, y], axis=1)


# ============================================================
# LINE
# ============================================================
def make_line_pool(key):
    k1, k2, k3 = jax.random.split(key, 3)

    L = jax.random.uniform(k1, (), minval=1.0, maxval=2.0)
    heading = jax.random.uniform(k2, (), minval=-jnp.pi, maxval=jnp.pi)

    margin = 0.2
    sx = jax.random.uniform(k3, (), minval=margin, maxval=POOL_X-margin)
    sy = jax.random.uniform(k3, (), minval=margin, maxval=POOL_Y-margin)

    t = jnp.linspace(0, L, N_PATH)

    x = sx + t*jnp.cos(heading)
    y = sy + t*jnp.sin(heading)

    x = jnp.clip(x, 0.0, POOL_X)
    y = jnp.clip(y, 0.0, POOL_Y)

    return jnp.stack([x, y], axis=1)


# ============================================================
# SINE
# ============================================================
def make_sine_pool(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)

    A = jax.random.uniform(k1, (), minval=1, maxval=2)
    lam = jax.random.uniform(k2, (), minval=2, maxval=2.5)

    x0 = jax.random.uniform(k3, (), minval=0.2, maxval=0.5)
    y_center = jax.random.uniform(k4, (), minval=0.6, maxval=1.4)

    x = jnp.linspace(x0, POOL_X-0.2, N_PATH)
    y = y_center + A*jnp.sin(2*jnp.pi*(x-x0)/lam)

    y = jnp.clip(y, 0.0, POOL_Y)

    return jnp.stack([x, y], axis=1)


# ============================================================
# RANDOM PATH
# ============================================================
def sample_path(key):
    k1, k2 = jax.random.split(key)
    choice = jax.random.randint(k1, (), 0, 3)

    def circle(_): return make_circle_pool(k2)
    def line(_):   return make_line_pool(k2)
    def sine(_):   return make_sine_pool(k2)

    return jax.lax.switch(choice, [circle, line, sine], operand=None)

def sample_paths_batch(key, N):
    keys = jax.random.split(key, N)
    paths = jax.vmap(sample_path)(keys)   # (N, N_path, 2)
    return paths


def plot_path(path):
    plt.figure(figsize=(6,4))
    plt.plot(path[:,0], path[:,1], 'o--',linewidth=3)

    plt.plot([0,POOL_X,POOL_X,0,0],
             [0,0,POOL_Y,POOL_Y,0],'k--')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == "__main__":
    key = jax.random.PRNGKey(10)

    for i in range(5):
        key, sub = jax.random.split(key)
        path = sample_path(sub)
        plot_path(path)

import jax
import jax.numpy as jnp

POOL_X = 3.0
POOL_Y = 2.0
N_PATH = 30


# ============================================================
# CIRCLE
# ============================================================
def make_circle_pool(key):
    k1, k2, k3 = jax.random.split(key, 3)

    r = jax.random.uniform(k1, (), minval=0.5, maxval=1)

    cx = jax.random.uniform(k2, (), minval=r, maxval=POOL_X-r)
    cy = jax.random.uniform(k3, (), minval=r, maxval=POOL_Y-r)

    theta = jnp.linspace(0, 2*jnp.pi, N_PATH)

    x = cx + r*jnp.cos(theta)
    y = cy + r*jnp.sin(theta)

    return jnp.stack([x, y], axis=1)

def make_semicircle_pool(key):
    k1, k2, k3 = jax.random.split(key, 3)

    r = jax.random.uniform(k1, (), minval=0.5, maxval=1.0)

    cx = jax.random.uniform(k2, (), minval=r, maxval=POOL_X-r)
    cy = jax.random.uniform(k3, (), minval=r, maxval=POOL_Y-r)

    theta = jnp.linspace(-jnp.pi/2, jnp.pi, N_PATH)  # forward arc

    x = cx + r*jnp.cos(theta)
    y = cy + r*jnp.sin(theta)

    return jnp.stack([x, y], axis=1)

# ============================================================
# LINE
# ============================================================
def make_line_pool(key):
    k1, k2, k3 = jax.random.split(key, 3)

    L = 15.0
    heading = jax.random.uniform(k2, (), minval=-jnp.pi, maxval=jnp.pi)

    margin = 0.5
    sx = jax.random.uniform(k3, (), minval=margin, maxval=POOL_X-margin)
    sy = jax.random.uniform(k3, (), minval=margin, maxval=POOL_Y-margin)

    t = jnp.linspace(0, L, N_PATH)

    x = sx + t*jnp.cos(heading)
    y = sy + t*jnp.sin(heading)

    # x = jnp.clip(x, 0.0, POOL_X)
    # y = jnp.clip(y, 0.0, POOL_Y)

    return jnp.stack([x, y], axis=1)


# ============================================================
# SINE
# ============================================================
def make_sine_pool(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)

    A = jax.random.uniform(k1, (), minval=0.5, maxval=1.2)
    lam = jax.random.uniform(k2, (), minval=2, maxval=6)

    x0 = jax.random.uniform(k3, (), minval=0.2, maxval=0.5)
    y_center = jax.random.uniform(k4, (), minval=0.6, maxval=1.4)

    x = jnp.linspace(x0, POOL_X+3, N_PATH)
    y = y_center + A*jnp.sin(2*jnp.pi*(x-x0)/lam)



    return jnp.stack([x, y], axis=1)


# ============================================================
# RANDOM PATH
# ============================================================
def sample_path(key):
    k1, k2 = jax.random.split(key)
    choice = jax.random.randint(k1, (), 0, 4)

    def circle(_): return make_circle_pool(k2)
    def line(_):   return make_line_pool(k2)
    def sine(_):   return make_sine_pool(k2)
    def semicircle(_): return make_semicircle_pool(k2)

    return jax.lax.switch(choice, [line,sine,semicircle], operand=None)

def sample_paths_batch(key, N):
    keys = jax.random.split(key, N)
    paths = jax.vmap(sample_path)(keys)   # (N, N_path, 2)
    return paths
# ============================================================
# helper
# ============================================================
def wrap_to_pi(a):
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


# ============================================================
# FIND CLOSEST PATH INDEX (vectorized)
# paths: (N_env, N_path, 2)
# robot_x/y: (N_env,)
# ============================================================
def closest_point_idx(paths, robot_x, robot_y):
    dx = paths[:,:,0] - robot_x[:,None]
    dy = paths[:,:,1] - robot_y[:,None]

    dist2 = dx*dx + dy*dy
    idx = jnp.argmin(dist2, axis=1)

    return idx


# ============================================================
# TANGENT FROM PATH
# ============================================================
def path_tangent(paths, idx):
    N_env, N_path, _ = paths.shape

    idx_next = jnp.clip(idx+1, 0, N_path-1)

    p1 = paths[jnp.arange(N_env), idx]
    p2 = paths[jnp.arange(N_env), idx_next]

    dx = p2[:,0] - p1[:,0]
    dy = p2[:,1] - p1[:,1]

    norm = jnp.sqrt(dx*dx + dy*dy) + 1e-8
    tx = dx/norm
    ty = dy/norm

    return tx, ty


# ============================================================
# CROSS TRACK ERROR
# ============================================================
def cross_track_error(paths, idx, robot_x, robot_y, tx, ty):
    N_env = paths.shape[0]

    proj = paths[jnp.arange(N_env), idx]

    nx = -ty
    ny = tx

    err = (robot_x - proj[:,0])*nx + (robot_y - proj[:,1])*ny
    return err


# ============================================================
# HEADING ERROR
# ============================================================
def heading_error(robot_heading, tx, ty):
    path_heading = jnp.arctan2(ty, tx)
    return wrap_to_pi(robot_heading - path_heading),path_heading


# ============================================================
# LOOKAHEAD POINTS
# returns k lookahead heading errors
# ============================================================
def lookahead_errors(paths, idx, robot_x, robot_y, robot_heading, Ls=[5,15,30,50]):
    """
    Ls = indices ahead along path (not meters)
    """

    N_env, N_path, _ = paths.shape
    errors = []

    for L in Ls:
        idx_la = jnp.clip(idx + L, 0, N_path-1)
        p_la = paths[jnp.arange(N_env), idx_la]

        dx = p_la[:,0] - robot_x
        dy = p_la[:,1] - robot_y

        psi_la = jnp.arctan2(dy, dx)
        err = wrap_to_pi(robot_heading - psi_la)
        errors.append(err)

    return jnp.stack(errors, axis=1)  # (N_env, num_lookahead)


# ============================================================
# MAIN FUNCTION (call every step)
# ============================================================
def compute_path_errors(paths, robot_x, robot_y, robot_heading):
    """
    paths: (N_env, N_path, 2)
    robot_x/y/heading: (N_env,)
    """

    idx = closest_point_idx(paths, robot_x, robot_y)

    tx, ty = path_tangent(paths, idx)

    ct_err = cross_track_error(paths, idx, robot_x, robot_y, tx, ty)
    hd_err,path_heading = heading_error(robot_heading, tx, ty)



    return ct_err, hd_err, path_heading, idx

def get_lookahead_indices(idx, N_path):
    L1, L2, L3 = 2, 4, 6

    i1 = (idx + L1) % N_path
    i2 = (idx + L2) % N_path
    i3 = (idx + L3) % N_path

    return i1, i2, i3
def get_lookahead_points_world(paths, idx):
    N_env, N_path, _ = paths.shape

    i1, i2, i3 = get_lookahead_indices(idx, N_path)

    p1 = paths[jnp.arange(N_env), i1]   # (N,2)
    p2 = paths[jnp.arange(N_env), i2]
    p3 = paths[jnp.arange(N_env), i3]

    return p1, p2, p3
def world_to_body(x, y, psi, pts):
    dx = pts[:,0] - x
    dy = pts[:,1] - y

    c = jnp.cos(psi)
    s = jnp.sin(psi)

    xb =  c*dx + s*dy
    yb = -s*dx + c*dy

    return jnp.stack([xb, yb], axis=1)  # (N,2)
def get_lookahead_body(paths, idx, xpos, ypos, qh):
    p1, p2, p3 = get_lookahead_points_world(paths, idx)

    b1 = world_to_body(xpos, ypos, qh, p1)
    b2 = world_to_body(xpos, ypos, qh, p2)
    b3 = world_to_body(xpos, ypos, qh, p3)

    return b1, b2, b3   # each (N,2)


def closest_point_idx(paths, robot_x, robot_y):
    dx = paths[:, :, 0] - robot_x[:, None]
    dy = paths[:, :, 1] - robot_y[:, None]
    dist2 = dx*dx + dy*dy
    idx = jnp.argmin(dist2, axis=1)
    return idx



import jax.numpy as jnp

def circle_lookahead(
    paths,
    robot_x,
    robot_y,
    last_idx,      # (N_env,) last visited path index
    L=0.3,
):
    """
    Pure pursuit:
    - circle intersection with path
    - choose MOST forward point along path
    - fallback = last visited index (never jump backward)

    paths: (N_env, N_path, 2)
    robot_x/y: (N_env,)
    last_idx: (N_env,)  int
    L: lookahead

    returns:
        target_pt (N_env,2)
        desired_heading (N_env,)
        new_last_idx (N_env,)
        found (N_env,)
    """

    N_env, N_path, _ = paths.shape
    N_seg = N_path - 1

    C = jnp.stack([robot_x, robot_y], axis=1)

    # segments
    P0 = paths[:, :-1, :]
    P1 = paths[:, 1:, :]
    d  = P1 - P0

    # only allow segments >= last_idx
    seg_ids = jnp.arange(N_seg)[None, :]
    valid_seg_mask = seg_ids >= last_idx[:, None]

    # quadratic solve
    f = P0 - C[:, None, :]
    a = jnp.sum(d*d, axis=2)
    b = 2.0 * jnp.sum(f*d, axis=2)
    c = jnp.sum(f*f, axis=2) - (L*L)[:, None]

    disc = b*b - 4*a*c
    has_real = disc >= 0

    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0))
    denom = 2*a + 1e-12

    t1 = (-b - sqrt_disc) / denom
    t2 = (-b + sqrt_disc) / denom

    valid1 = has_real & (t1 >= 0) & (t1 <= 1) & valid_seg_mask
    valid2 = has_real & (t2 >= 0) & (t2 <= 1) & valid_seg_mask

    p1 = P0 + t1[:,:,None]*d
    p2 = P0 + t2[:,:,None]*d

    # forward along path
    seg_len = jnp.sqrt(a) + 1e-12
    seg_dir = d / seg_len[:,:,None]

    prog1 = jnp.sum((p1 - P0)*seg_dir, axis=2)
    prog2 = jnp.sum((p2 - P0)*seg_dir, axis=2)

    valid1 = valid1 & (prog1 > 0)
    valid2 = valid2 & (prog2 > 0)

    neg_inf = -1e30
    score1 = jnp.where(valid1, prog1, neg_inf)
    score2 = jnp.where(valid2, prog2, neg_inf)

    pick2 = score2 > score1
    best_score_seg = jnp.where(pick2, score2, score1)
    best_pt_seg = jnp.where(pick2[:,:,None], p2, p1)

    best_seg_idx = jnp.argmax(best_score_seg, axis=1)
    best_score = best_score_seg[jnp.arange(N_env), best_seg_idx]

    found = best_score > (neg_inf/2)

    target_pt = best_pt_seg[jnp.arange(N_env), best_seg_idx]

    # fallback → last visited index
    fb_pt = paths[jnp.arange(N_env), last_idx]

    target_pt = jnp.where(found[:,None], target_pt, fb_pt)

    # update last visited index
    new_last_idx = jnp.where(found, best_seg_idx, last_idx)

    desired_heading = jnp.arctan2(
        target_pt[:,1] - robot_y,
        target_pt[:,0] - robot_x
    )
    desired_heading = jnp.arctan2(jnp.sin(desired_heading), jnp.cos(desired_heading))

    return target_pt, desired_heading, new_last_idx, found


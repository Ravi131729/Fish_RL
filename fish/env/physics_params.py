import jax.numpy as jnp
import chex

import chex
import jax.numpy as jnp

@chex.dataclass
class PhysicsParams:
    b1: jnp.ndarray
    bs: jnp.ndarray
    l11: jnp.ndarray
    l21: jnp.ndarray
    ls1: jnp.ndarray
    c1: jnp.ndarray

    added_mass_scale: jnp.ndarray
    inertia_scale: jnp.ndarray
    head_damping_scale: jnp.ndarray
    link_damping_scale: jnp.ndarray
    stiffness_scale: jnp.ndarray



def get_constants(p: PhysicsParams) -> jnp.ndarray:
    rho = 997.0
    AM  = 1.0

    b1  = p.b1
    bs  = p.bs
    l11 = p.l11
    l21 = p.l21
    ls1 = p.ls1
    c1  = p.c1

    # --- head ---
    m_h = 0.44
    I_h = m_h * (b1**2 + bs**2) / 4

    ma_hx1 = (m_h + AM*jnp.pi*rho*(bs**2)*b1) * p.added_mass_scale
    ma_hy1 = (m_h + AM*jnp.pi*rho*(b1**3))    * p.added_mass_scale
    Ia_h1  = (I_h + AM*(1/8)*jnp.pi*rho*b1*(b1**2 - bs**2)**2) * p.inertia_scale

    # --- links ---
    m_l = 0.01 / 2

    ma_l1x1 = m_l * p.added_mass_scale
    ma_l1y1 = (m_l + AM*jnp.pi*rho*0.075*(l11/2)**2) * p.added_mass_scale

    ma_l2x1 = m_l * p.added_mass_scale
    ma_l2y1 = (m_l + AM*jnp.pi*rho*0.075*(l21/2)**2) * p.added_mass_scale

    # --- short link ---
    m_ls = 0.01
    ma_lsx1 = m_ls * p.added_mass_scale
    ma_lsy1 = (m_ls + AM*jnp.pi*rho*0.075*(ls1/2)**2) * p.added_mass_scale

    # --- inertias ---
    Ia_l11 = ((1/12) * m_l * l11**2) * p.inertia_scale
    Ia_l21 = ((1/12) * m_l * l21**2) * p.inertia_scale
    Ia_ls1 = ((1/12) * m_ls * ls1**2) * p.inertia_scale
    I_r1   = (0.1 * 0.027**2) * p.inertia_scale

    # --- damping ---
    C_hx1 = 0.46 * p.head_damping_scale
    C_hy1 = 10.0 * p.head_damping_scale
    C_lx1 = 0.0
    C_ly1 = 10.0 * p.link_damping_scale

    # --- stiffness ---
    K_11 = 0.4 * p.stiffness_scale
    K_21 = 0.7 * p.stiffness_scale

    Length = (l11, l21, ls1, b1, c1)
    AddedMass = (ma_l1x1, ma_l1y1, ma_l2x1, ma_l2y1, ma_hx1, ma_hy1, ma_lsx1, ma_lsy1)
    AddedInertia = (Ia_h1, Ia_l11, Ia_l21, Ia_ls1, I_r1)
    Dissipation = (C_hx1, C_hy1, C_lx1, C_ly1)
    Stiffness = (K_11, K_21)

    return jnp.array((*Length, *AddedMass, *AddedInertia, *Dissipation, *Stiffness), dtype=jnp.float32)

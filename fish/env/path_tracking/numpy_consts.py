import numpy as np

def get_constants():
    rho = 997   # density
    AM = 1.0  # activate added mass coefficient

    # For Head link
    b1 = 0.075                                                       # length major axis (m)
    bs = 0.035                                                      # length minor axis (m)
    m_h = 0.44                                                      # mass (kg)
    I_h = m_h * (b1**2 + bs**2) / 4                                  # m.o.i (Kg/m^2)
    ma_hx1 = m_h + AM*np.pi*rho * (bs**2 )* b1                        # added mass (kg)
    ma_hy1 = m_h + AM*np.pi * rho * (b1)**3                           # added mass (kg)
    Ia_h1 = I_h + AM*(1/8) * np.pi * rho * b1 * (b1**2 - bs**2)**2    # added m.o.i (Kg/m^2)

    # For link1
    l11 = 0.048                                                      # length (m)
    m_l1 = 0.01 / 2                                                 # mass (kg)
    I_l1 = (1/12) * m_l1 * l11**2                                    # m.o.i (Kg/m^2)
    ma_l1x1 = m_l1 + AM*0                                                   # added mass (kg)
    ma_l1y1 = m_l1 + AM*np.pi * rho * 0.075 * (l11 / 2)**2            # added mass (kg)
    Ia_l11 = I_l1 + AM*(1/8) * np.pi * rho * 0.075 * (l11 / 2)**4    # added m.o.i (Kg/m^2)

    # For link2
    l21 = 0.048                                                      # length (m)
    m_l2 = 0.01 / 2                                                 # mass (kg)
    I_l2 = (1/12) * m_l2 * l21**2                                    # m.o.i (Kg/m^2)
    ma_l2x1 = m_l2 + AM*0                                                   # added mass (kg)
    ma_l2y1 = m_l2 + AM*np.pi * rho * 0.075 * (l21 / 2)**2            # added mass (kg)
    Ia_l21 = I_l2 + AM*(1/8) * np.pi * rho * 0.075 * (l21 / 2)**4    # added m.o.i (Kg/m^2)

    # For short link
    ls1 = 0.015                                                      # length (m)
    m_ls = 0.01                                                     # mass (kg)
    I_ls = (1/12) * m_ls * ls1**2                                    # m.o.i (Kg/m^2)
    ma_lsx1 = m_ls + AM*0                                                    # added mass (kg)
    ma_lsy1 = m_ls + AM*np.pi * rho * 0.075 * (ls1 / 2)**2            # added mass (kg)
    Ia_ls1 = I_ls + AM*(1/8) * np.pi * rho * 0.075 * (ls1 / 2)**4    # added m.o.i (Kg/m^2)

    # For rotor
    c1 = 0.03                                   # length (m)
    m_r = 0.1                                  # mass (kg)
    I_r1 = m_r * 0.027**2                        # m.o.i (Kg/m^2)

    # Constant Rayleigh dissipation
    C_hx1 = 0.46
    C_hy1 = 10
    C_lx1 = 0
    C_ly1 = 10
    K_11 = 0.4           # stiffness (Nm/rad)
    K_21 = 0.7           # stiffness (Nm/rad)

    Length = (l11, l21, ls1, b1, c1)
    Stiffness = ( K_11, K_21)
    AddedMass = (ma_l1x1, ma_l1y1, ma_l2x1, ma_l2y1, ma_hx1, ma_hy1, ma_lsx1, ma_lsy1)
    AddedInertia = (Ia_h1, Ia_l11, Ia_l21, Ia_ls1, I_r1)
    Dissipation = (C_hx1, C_hy1, C_lx1, C_ly1)
    Const = (*Length, *AddedMass, *AddedInertia, *Dissipation, *Stiffness)
    return np.array(Const)

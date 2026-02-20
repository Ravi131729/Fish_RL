
from numpy.linalg import solve
from numpy_matrices import mass_matrix, coriolis_vector, gravity_vector
from numpy_consts import get_constants
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

# jax.config.update("jax_platform_name", "cpu")
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
    return np.array([x[3],x[4],x[0],x[5],x[1],x[6],x[2]])
def f(x, inp, const_vals):
    """
    x : state vector [q1, q2, qh, u, qd1,  qd2, qdh]
    u : input vector [alpha, dalpha, ddalpha, ddphi]
    """
    q_dot = np.array([x[4],x[5],x[6]])
    states = get_ordered_states(x)
    return np.concatenate([q_dot,
                     dynamics(states, inp, const_vals)])

def rk4_step(x, inp, const_vals, dt):
    k1 = f(x, inp, const_vals)
    k2 = f(x + 0.5 * dt * k1, inp, const_vals)
    k3 = f(x + 0.5 * dt * k2, inp, const_vals)
    k4 = f(x + dt * k3, inp, const_vals)

    return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)




# I = 90e-5
# A = 10/I
# w = 2
# omega = 2*np.pi*w
# dt = 0.001
# T  = 10.0
# N  = int(T / dt)


# x = np.zeros(7)
# const_vals = get_constants()



# # Storage
# xs = np.zeros((N, len(x)))
# ts = np.zeros(N)
# deltas = np.zeros(N)


# x = np.zeros(7)
# x_pos, y_pos = 0.0, 0.0  #
# X = np.zeros(N)
# Y = np.zeros(N)

# for k in range(N):
#     t = k * dt

#     # --- extract states ---
#     qh = x[2]
#     r  = x[6]
#     u  = x[3]

#     # --- world-frame velocity ---
#     xd = u * np.cos(qh)
#     yd = u * np.sin(qh)

#     # --- integrate position ---
#     x_pos += xd * dt
#     y_pos += yd * dt

#     A_t = A * (1.0 - np.exp(-t / 0.1))
#     delta = 1.48
#     inp = np.array([
#         delta,
#         0.0,
#         0.0,
#         -A_t * np.sin(2.0 * np.pi * w * t)
#     ])

#     # --- integrate dynamics ---
#     x = rk4_step(x, inp, const_vals, dt)

#     X[k] = x_pos
#     Y[k] = y_pos
#     xs[k] = x
#     ts[k] = t
# x = xs
# t =ts
# q1  = x[:, 0]
# q2  = x[:, 1]
# qh  = x[:, 2]
# u   = x[:, 3]
# qd1 = x[:, 4]
# qd2 = x[:, 5]
# qdh = x[:, 6]
# xd = u * np.cos(qh)
# yd = u * np.sin(qh)

# l1 = l2 = 0.048
# ls1 = 0.015
# b=0.075*1.5
# xc = X + l1*np.cos(q1) + l2*np.cos(q2) +ls1*np.cos(qh - 0.0) + b*np.cos(qh)
# yc =  Y + l1*np.sin(q1) + l2*np.sin(q2) +ls1*np.sin(qh -0.0) + b*np.sin(qh)
# xcd = (xc[1:] - xc[:-1])/dt
# ycd = (yc[1:] - yc[:-1])/dt
# ub = xcd*np.cos(qh[:-1]) + ycd*np.sin(qh[:-1])
# import matplotlib.pyplot as plt

# plt.figure()
# # plt.plot(X, Y, label='Tail Position')
# plt.plot(xc, yc, label='Head Position')
# plt.axis('equal')
# plt.xlabel('X Position (m)')
# plt.ylabel('Y Position (m)')
# plt.title('Fish Trajectory')
# plt.legend()
# plt.grid()
# plt.show()

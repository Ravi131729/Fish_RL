"""
Path Tracking Main Script

Run fish robot path tracking simulation with Stanley controller.

Usage:
    python -m path_tracking.main --path sinusoidal --time 15
    python -m path_tracking.main --path circular --time 20
    python -m path_tracking.main --path figure_eight --time 30
    python -m path_tracking.main --path random --time 25
    python -m path_tracking.main --path random --time 25 --seed 42  # Reproducible random path
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from controllers import StanleyController

from dynamics import rk4_step, compute_head_position, wrap_to_pi
from paths import sinusoidal_path, straight_path, circular_path, figure_eight_path, square_path, RandomClosedPath

# Import constants from parent directory
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from numpy_consts import get_constants


def run_stanley_tracking(path_type='sinusoidal', T=15.0, dt=0.01, seed=None):
    """
    Run fish simulation with Stanley controller for path tracking.

    Args:
        path_type: 'sinusoidal', 'straight', 'circular', 'figure_eight', 'square', or 'random'
        T: Simulation time (seconds)
        dt: Time step (seconds)
        seed: Random seed for 'random' path type (None for different path each run)

    Returns:
        Dictionary with simulation results
    """
    # Simulation parameters
    I = 90e-5
    A = 8.0 / I  # Propulsion amplitude
    w = 3         # Propulsion frequency (Hz)
    N = int(T / dt)

    # Stanley controller - tuned for each path type
    if path_type == 'circular':
        stanley = StanleyController(k=1.0, k_soft=0.001, max_steer=np.pi/3)
    elif path_type == 'figure_eight':
        stanley = StanleyController(k=1.5, k_soft=0.001, max_steer=np.pi/3)
    elif path_type == 'square':
        stanley = StanleyController(k=0.5, k_soft=0.3, max_steer=np.pi/3)
    elif path_type == 'random':
        stanley = StanleyController(k=0.5, k_soft=0.3, max_steer=np.pi/3)
    else:
        stanley = StanleyController(k=0.20, k_soft=0.002, max_steer=np.pi/3)

    # Set timestep for rate limiting
    stanley.set_dt(dt)

    # Path parameters based on type
    path_params = {}
    random_path_obj = None  # For random path
    if path_type == 'sinusoidal':
        path_params = {
            'amplitude': 1.0,
            'wavelength': 2,
            'phase': -np.pi/2,
            'y_offset': 0.15
        }
    elif path_type == 'straight':
        path_params = {
            'angle': np.deg2rad(60),
            'y_offset': 0.0
        }
    elif path_type == 'circular':
        path_params = {
            'radius': 0.5,
            'center_x': 1.5,
            'center_y': 0.0
        }
    elif path_type == 'figure_eight':
        path_params = {
            'scale': 1.5,
            'y_offset': 1.0
        }
    elif path_type == 'square':
        path_params = {
            'side_length': 2.0,
            'center_x': 1.0,
            'center_y': 1.0
        }
    elif path_type == 'random':
        # Generate random smooth closed path
        # Use seed=None for different path each run, or set seed for reproducibility
        random_path_obj = RandomClosedPath(
            n_points=8,           # Number of control points
            base_radius=2.0,      # Base radius
            randomness=0.6,       # Perturbation amount (0-1)
            center_x=2.0,
            center_y=1.5,
            seed=seed             # None = random each time, or set integer for reproducibility
        )

    # Get physics constants
    const_vals = get_constants()

    # Initialize state
    x = np.zeros(7)  # [q1, q2, qh, u, qd1, qd2, qdh]
    tail_xpos, tail_ypos = 0.0, 0.0

    # Storage arrays
    xs = np.zeros((N, 7))
    ts = np.zeros(N)
    deltas = np.zeros(N)
    head_X = np.zeros(N)
    head_Y = np.zeros(N)
    path_Y_ref = np.zeros(N)
    path_psi_ref = np.zeros(N)
    errors_ct = np.zeros(N)
    errors_psi = np.zeros(N)

    # Heading averaging for smoother control
    heading_history = []
    heading_avg_window = 1

    print(f"Running STANLEY controller with {path_type.upper()} path...")
    print(f"Simulation time: {T}s, dt: {dt}s")

    for k in range(N):
        t = k * dt
        qh = x[2]
        u = x[3]

        # Average heading (circular mean)
        heading_history.append(qh)
        if len(heading_history) > heading_avg_window:
            heading_history.pop(0)
        sin_avg = np.mean([np.sin(h) for h in heading_history])
        cos_avg = np.mean([np.cos(h) for h in heading_history])
        qh_avg = np.arctan2(sin_avg, cos_avg)

        # Current head position
        head_x, head_y = compute_head_position(x, tail_xpos, tail_ypos,
                                                deltas[k-1] if k > 0 else 0.0)

        # Get reference based on path type
        y_ref, psi_ref, e_ct = 0.0, 0.0, 0.0  # Default values
        if path_type == 'sinusoidal':
            y_ref, psi_ref = sinusoidal_path(head_x, **path_params)
            e_ct = head_y - y_ref
        elif path_type == 'straight':
            y_ref, psi_ref = straight_path(head_x, **path_params)
            e_ct = head_y - y_ref
        elif path_type == 'circular':
            y_ref, psi_ref, e_ct = circular_path(head_x, head_y, **path_params)
        elif path_type == 'figure_eight':
            y_ref, psi_ref, e_ct = figure_eight_path(head_x, head_y, **path_params)
        elif path_type == 'square':
            y_ref, psi_ref, e_ct = square_path(head_x, head_y, **path_params)
        elif path_type == 'random':
            y_ref, psi_ref, e_ct = random_path_obj.get_reference(head_x, head_y)

        # Stanley control
        delta = stanley.compute(qh_avg, psi_ref, e_ct, u)

        if np.isnan(delta):
            delta = 0.0

        # Propulsion input with slow ramp-up for stability
        A_t = A * (1.0 - np.exp(-t / 0.5))
        inp = np.array([delta, 0.0, 0.0, -A_t * np.sin(2.0 * np.pi * w * t)])

        # Store data
        xs[k] = x
        ts[k] = t
        deltas[k] = delta
        head_X[k], head_Y[k] = head_x, head_y
        path_Y_ref[k] = y_ref
        path_psi_ref[k] = psi_ref
        errors_ct[k] = e_ct
        errors_psi[k] = wrap_to_pi(qh_avg - psi_ref)

        # Integrate dynamics
        x_new = rk4_step(x, inp, const_vals, dt)

        # Check for instability
        if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
            print(f"Unstable at t={t:.2f}s")
            # Truncate arrays
            xs = xs[:k]
            ts = ts[:k]
            deltas = deltas[:k]
            head_X = head_X[:k]
            head_Y = head_Y[:k]
            path_Y_ref = path_Y_ref[:k]
            path_psi_ref = path_psi_ref[:k]
            errors_ct = errors_ct[:k]
            errors_psi = errors_psi[:k]
            break

        x = x_new

        # Update tail position
        tail_xpos += u * np.cos(qh) * dt
        tail_ypos += u * np.sin(qh) * dt

    # Print results
    print(f"Final head position: ({head_X[-1]:.3f}, {head_Y[-1]:.3f})")
    print(f"Mean cross-track error: {np.mean(np.abs(errors_ct)):.4f} m")
    print(f"Mean heading error: {np.mean(np.abs(errors_psi)) * 180/np.pi:.2f} deg")

    return {
        'ts': ts,
        'xs': xs,
        'deltas': deltas,
        'head_X': head_X,
        'head_Y': head_Y,
        'path_Y_ref': path_Y_ref,
        'path_psi_ref': path_psi_ref,
        'errors_ct': errors_ct,
        'errors_psi': errors_psi,
        'path_type': path_type,
        'path_params': path_params,
        'random_path_obj': random_path_obj,
        'A': A,
        'w': w
    }


def plot_results(results):
    """
    Plot simulation results.

    Args:
        results: Dictionary from run_stanley_tracking()
    """
    ts = results['ts']
    xs = results['xs']
    deltas = results['deltas']
    head_X = results['head_X']
    head_Y = results['head_Y']
    path_psi_ref = results['path_psi_ref']
    errors_ct = results['errors_ct']
    errors_psi = results['errors_psi']
    path_type = results['path_type']
    path_params = results['path_params']
    random_path_obj = results.get('random_path_obj', None)
    controller_name = results.get('controller', 'stanley').capitalize()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Path tracking (XY plane)
    ax1 = axes[0, 0]

    # Generate reference path for plotting
    if path_type == 'sinusoidal':
        x_plot = np.linspace(min(head_X)-0.1, max(head_X)+0.1, 200)
        y_plot, _ = sinusoidal_path(x_plot, **path_params)
        ax1.plot(x_plot, y_plot, 'r--', linewidth=2, label='Reference Path')
    elif path_type == 'straight':
        x_plot = np.linspace(min(head_X)-0.1, max(head_X)+0.1, 200)
        y_plot, _ = straight_path(x_plot, **path_params)
        ax1.plot(x_plot, y_plot, 'r--', linewidth=2, label='Reference Path')
    elif path_type == 'circular':
        theta_plot = np.linspace(0, 2*np.pi, 200)
        x_plot = path_params['center_x'] + path_params['radius'] * np.cos(theta_plot)
        y_plot = path_params['center_y'] + path_params['radius'] * np.sin(theta_plot)
        ax1.plot(x_plot, y_plot, 'r--', linewidth=2, label='Reference Path')
    elif path_type == 'figure_eight':
        t_plot = np.linspace(0, 2*np.pi, 500)
        a = path_params['scale']
        x_plot = a * np.cos(t_plot) / (1 + np.sin(t_plot)**2)
        y_plot = a * np.sin(t_plot) * np.cos(t_plot) / (1 + np.sin(t_plot)**2) + path_params['y_offset']
        ax1.plot(x_plot, y_plot, 'r--', linewidth=2, label='Reference Path')
    elif path_type == 'square':
        half = path_params['side_length'] / 2.0
        cx, cy = path_params['center_x'], path_params['center_y']
        # Square corners (closed loop)
        x_plot = [cx - half, cx + half, cx + half, cx - half, cx - half]
        y_plot = [cy - half, cy - half, cy + half, cy + half, cy - half]
        ax1.plot(x_plot, y_plot, 'r--', linewidth=2, label='Reference Path')
    elif path_type == 'random':
        x_plot, y_plot = random_path_obj.get_path_points(300)
        ax1.plot(x_plot, y_plot, 'r--', linewidth=2, label='Reference Path')

    ax1.plot(head_X, head_Y, 'b-', linewidth=1.5, label='Fish Trajectory')
    ax1.plot(head_X[0], head_Y[0], 'go', markersize=10, label='Start')
    ax1.plot(head_X[-1], head_Y[-1], 'r*', markersize=15, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'{controller_name} Controller - {path_type.capitalize()} Path')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True)

    # Plot 2: Cross-track error
    ax2 = axes[0, 1]
    ax2.plot(ts, errors_ct * 100, 'b-')
    ax2.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Cross-track Error (cm)')
    ax2.set_title('Cross-track Error')
    ax2.grid(True)

    # Plot 3: Heading error
    ax3 = axes[0, 2]
    ax3.plot(ts, np.rad2deg(errors_psi), 'g-')
    ax3.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Heading Error (deg)')
    ax3.set_title('Heading Error')
    ax3.grid(True)

    # Plot 4: Steering angle
    ax4 = axes[1, 0]
    ax4.plot(ts, np.rad2deg(deltas), 'purple')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Delta (deg)')
    ax4.set_title('Steering Angle')
    ax4.grid(True)

    # Plot 5: Forward velocity
    ax5 = axes[1, 1]
    ax5.plot(ts, xs[:, 3], 'orange')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Velocity (m/s)')
    ax5.set_title('Forward Velocity')
    ax5.grid(True)

    # Plot 6: Heading comparison
    ax6 = axes[1, 2]
    ax6.plot(ts, np.rad2deg(xs[:, 2]), 'cyan', label='Fish Heading')
    ax6.plot(ts, np.rad2deg(path_psi_ref), 'r--', label='Path Tangent')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Heading (deg)')
    ax6.set_title('Heading Comparison')
    ax6.legend()
    ax6.grid(True)

    A = results.get('A', 0)
    A = A*90e-5
    w = results.get('w', 0)
    fig.suptitle(f'Stanley Controller - {path_type.capitalize()} Path (A={A:.0f}, w={w})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'stanley_{path_type}_A{A:.0f}_w{w}_tracking.png', dpi=150)
    plt.show()


def main():
    """Main entry point for path tracking simulation."""
    parser = argparse.ArgumentParser(description='Fish Robot Path Tracking')
    parser.add_argument('--path', type=str, default='sinusoidal',
                        choices=['sinusoidal', 'straight', 'circular', 'figure_eight', 'square', 'random'],
                        help='Path type to track')
    parser.add_argument('--time', type=float, default=15.0,
                        help='Simulation time (seconds)')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step (seconds)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for random path (for reproducibility)')

    args = parser.parse_args()

    print("="*60)
    print("STANLEY Path Tracking Controller for Fish Robot")
    print("="*60)

    results = run_stanley_tracking(path_type=args.path, T=args.time, dt=args.dt, seed=args.seed)

    if not args.no_plot:
        plot_results(results)

    return results


if __name__ == "__main__":
    main()
"""
Path Definitions

Contains various reference path functions for path tracking.
Each path function returns the reference y-position, heading angle (psi_ref),
and optionally the cross-track error.
"""

import numpy as np
from scipy.interpolate import splprep, splev


class RandomClosedPath:
    """
    Generates a random smooth closed path using cubic spline interpolation.

    The path is created by:
    1. Generating random control points around a circle
    2. Adding random radial perturbations
    3. Fitting a smooth closed spline through the points
    """

    def __init__(self, n_points=10, base_radius=1.5, randomness=0.5,
                 center_x=0.0, center_y=1.5, seed=None):
        """
        Initialize random closed path.

        Args:
            n_points: Number of control points (more = more complex shape)
            base_radius: Base radius of the path
            randomness: Amount of random perturbation (0-1, fraction of base_radius)
            center_x: Center x coordinate
            center_y: Center y coordinate
            seed: Random seed for reproducibility (None for different path each time)
        """
        self.center_x = center_x
        self.center_y = center_y
        self.seed = seed

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Generate random control points around a circle
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)

        # Add some angular perturbation for more natural shapes
        angles += np.random.uniform(-0.2, 0.2, n_points)
        angles = np.sort(angles)  # Keep them ordered

        # Random radii with perturbation
        radii = base_radius * (1 + randomness * np.random.uniform(-1, 1, n_points))

        # Convert to Cartesian coordinates
        ctrl_x = center_x + radii * np.cos(angles)
        ctrl_y = center_y + radii * np.sin(angles)

        # Close the loop by appending first point
        ctrl_x = np.append(ctrl_x, ctrl_x[0])
        ctrl_y = np.append(ctrl_y, ctrl_y[0])

        # Fit a smooth closed spline
        # per=True makes it periodic (closed)
        tck, u = splprep([ctrl_x, ctrl_y], s=0, per=True, k=3)
        self.tck = tck

        # Pre-compute dense path points for fast closest point lookup
        self.n_lookup = 500
        u_dense = np.linspace(0, 1, self.n_lookup)
        self.path_x, self.path_y = splev(u_dense, tck)

        # Pre-compute tangents
        dx, dy = splev(u_dense, tck, der=1)
        self.tangent_angles = np.arctan2(dy, dx)

        # Normalize tangent vectors
        tang_len = np.sqrt(dx**2 + dy**2)
        self.tang_x = dx / tang_len
        self.tang_y = dy / tang_len

    def get_reference(self, x, y):
        """
        Get reference point, heading, and cross-track error for current position.

        Args:
            x: Current x position
            y: Current y position

        Returns:
            (y_ref, psi_ref, e_ct): Reference y, tangent angle, and cross-track error
        """
        # Find closest point on path using pre-computed lookup
        dist_sq = (self.path_x - x)**2 + (self.path_y - y)**2
        idx = np.argmin(dist_sq)

        x_ref = self.path_x[idx]
        y_ref = self.path_y[idx]
        psi_ref = self.tangent_angles[idx]

        # Cross-track error: signed distance (positive = left of path direction)
        to_fish_x = x - x_ref
        to_fish_y = y - y_ref
        e_ct = self.tang_x[idx] * to_fish_y - self.tang_y[idx] * to_fish_x

        return y_ref, psi_ref, e_ct

    def get_path_points(self, n_points=200):
        """
        Get dense path points for visualization.

        Args:
            n_points: Number of points to return

        Returns:
            (x_path, y_path): Arrays of x and y coordinates
        """
        u = np.linspace(0, 1, n_points)
        x_path, y_path = splev(u, self.tck)
        return x_path, y_path


def random_closed_path(x, y, path_obj):
    """
    Wrapper function to use RandomClosedPath with the same interface as other paths.

    Args:
        x: Current x position
        y: Current y position
        path_obj: RandomClosedPath instance

    Returns:
        (y_ref, psi_ref, e_ct): Reference y, tangent angle, and cross-track error
    """
    return path_obj.get_reference(x, y)


def square_path(x, y, side_length=2.0, center_x=1.0, center_y=1.0):
    """
    Square path centered at (center_x, center_y) with given side length.
    The square has corners at:
        - Bottom-left: (center_x - side/2, center_y - side/2)
        - Bottom-right: (center_x + side/2, center_y - side/2)
        - Top-right: (center_x + side/2, center_y + side/2)
        - Top-left: (center_x - side/2, center_y + side/2)

    Path direction: Counter-clockwise starting from bottom edge.

    Args:
        x: Current x position
        y: Current y position
        side_length: Length of each side of the square
        center_x: Square center x
        center_y: Square center y

    Returns:
        (y_ref, psi_ref, e_ct): Reference y, tangent angle, and cross-track error
    """
    half = side_length / 2.0

    # Square corners (CCW order starting from bottom-left)
    corners = [
        (center_x - half, center_y - half),  # Bottom-left
        (center_x + half, center_y - half),  # Bottom-right
        (center_x + half, center_y + half),  # Top-right
        (center_x - half, center_y + half),  # Top-left
    ]

    # Four edges with their directions (CCW)
    edges = [
        (corners[0], corners[1], 0.0),           # Bottom: moving right, psi = 0
        (corners[1], corners[2], np.pi/2),       # Right: moving up, psi = pi/2
        (corners[2], corners[3], np.pi),         # Top: moving left, psi = pi
        (corners[3], corners[0], -np.pi/2),      # Left: moving down, psi = -pi/2
    ]

    # Find closest point on the square
    best_dist = float('inf')
    best_x_ref = corners[0][0]
    best_y_ref = corners[0][1]
    best_psi_ref = 0.0
    best_e_ct = 0.0

    for (p1, p2, psi) in edges:
        x1, y1 = p1
        x2, y2 = p2

        # Edge vector
        ex = x2 - x1
        ey = y2 - y1
        edge_len = np.sqrt(ex**2 + ey**2)

        if edge_len < 1e-6:
            continue

        # Normalize edge vector
        ex_n = ex / edge_len
        ey_n = ey / edge_len

        # Vector from p1 to current position
        px = x - x1
        py = y - y1

        # Project onto edge
        proj = px * ex_n + py * ey_n
        proj = np.clip(proj, 0, edge_len)

        # Closest point on this edge
        x_closest = x1 + proj * ex_n
        y_closest = y1 + proj * ey_n

        # Distance to closest point
        dist = np.sqrt((x - x_closest)**2 + (y - y_closest)**2)

        if dist < best_dist:
            best_dist = dist
            best_x_ref = x_closest
            best_y_ref = y_closest
            best_psi_ref = psi

            # Cross-track error: positive = left of path direction
            # Use cross product: tangent x (fish - ref)
            to_fish_x = x - x_closest
            to_fish_y = y - y_closest
            # Cross product (2D): tangent_x * to_fish_y - tangent_y * to_fish_x
            best_e_ct = ex_n * to_fish_y - ey_n * to_fish_x

    return best_y_ref, best_psi_ref, best_e_ct


def sinusoidal_path(x, amplitude=0.3, wavelength=2.0, phase=0.0, y_offset=0.0):
    """
    Sinusoidal path: y = amplitude * sin(2*pi*x/wavelength + phase) + y_offset

    Args:
        x: Current x position
        amplitude: Wave amplitude
        wavelength: Wave wavelength
        phase: Phase offset
        y_offset: Vertical offset

    Returns:
        (y_ref, psi_ref): Reference y position and tangent angle
    """
    k = 2 * np.pi / wavelength
    y_ref = amplitude * np.sin(k * x + phase) + y_offset

    # Tangent angle (derivative of y w.r.t x)
    dy_dx = amplitude * k * np.cos(k * x + phase)
    psi_ref = np.arctan(dy_dx)

    return y_ref, psi_ref


def straight_path(x, y_offset=0.0, angle=0.0):
    """
    Straight line path: y = y_offset + x * tan(angle)

    Args:
        x: Current x position
        y_offset: Y-intercept
        angle: Path angle (radians)

    Returns:
        (y_ref, psi_ref): Reference y position and heading angle
    """
    y_ref = y_offset + x * np.tan(angle)
    psi_ref = angle
    return y_ref, psi_ref


def circular_path(x, y, radius=1.0, center_x=0.0, center_y=1.0):
    """
    Circular path centered at (center_x, center_y) with given radius.
    Returns closest point on circle and tangent angle for CCW motion.

    Cross-track error convention: positive = to the LEFT of path direction
    For CCW circle: LEFT of path = OUTSIDE the circle

    Args:
        x: Current x position
        y: Current y position
        radius: Circle radius
        center_x: Circle center x
        center_y: Circle center y

    Returns:
        (y_ref, psi_ref, e_ct): Reference y, tangent angle, and cross-track error
    """
    # Vector from center to current position
    dx = x - center_x
    dy = y - center_y
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 1e-6:
        dist = 1e-6

    # Closest point on circle
    x_ref = center_x + radius * dx / dist
    y_ref = center_y + radius * dy / dist

    # Tangent angle for CCW motion (perpendicular to radius, 90 deg CCW from radial)
    psi_ref = np.arctan2(dx, -dy)

    # Cross-track error:
    # e_ct = radius - dist: positive = inside = right of path for CCW
    e_ct = radius - dist

    return y_ref, psi_ref, e_ct


def figure_eight_path(x, y, scale=2.0, y_offset=1.0):
    """
    Figure-eight (lemniscate of Bernoulli) path.

    Parametric form:
        x = scale * cos(t) / (1 + sin²(t))
        y = scale * sin(t) * cos(t) / (1 + sin²(t)) + y_offset

    Args:
        x: Current x position
        y: Current y position
        scale: Size scale of the figure-eight
        y_offset: Vertical offset

    Returns:
        (y_ref, psi_ref, e_ct): Reference y, tangent angle, and cross-track error
    """
    # Find the parameter t that gives the closest point on the figure-eight
    theta_guess = np.arctan2(y - y_offset, x)

    # Search for closest point
    best_t = theta_guess
    best_dist = float('inf')

    # Coarse search over parameter t
    for t in np.linspace(-np.pi, np.pi, 100):
        denom = 1 + np.sin(t)**2
        x_path = scale * np.cos(t) / denom
        y_path = scale * np.sin(t) * np.cos(t) / denom + y_offset
        dist = (x - x_path)**2 + (y - y_path)**2
        if dist < best_dist:
            best_dist = dist
            best_t = t

    # Refine with finer search around best_t
    for t in np.linspace(best_t - 0.1, best_t + 0.1, 50):
        denom = 1 + np.sin(t)**2
        x_path = scale * np.cos(t) / denom
        y_path = scale * np.sin(t) * np.cos(t) / denom + y_offset
        dist = (x - x_path)**2 + (y - y_path)**2
        if dist < best_dist:
            best_dist = dist
            best_t = t

    # Compute reference point and tangent at best_t
    t = best_t
    denom = 1 + np.sin(t)**2
    x_ref = scale * np.cos(t) / denom
    y_ref = scale * np.sin(t) * np.cos(t) / denom + y_offset

    # Tangent: derivative of parametric curve (analytically computed)
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    denom2 = denom**2

    dx_dt = scale * (-sin_t * denom - cos_t * 2 * sin_t * cos_t) / denom2
    dy_dt = scale * ((cos_t**2 - sin_t**2) * denom - sin_t * cos_t * 2 * sin_t * cos_t) / denom2

    psi_ref = np.arctan2(dy_dt, dx_dt)

    # Cross-track error: signed distance to path (positive = left of path direction)
    to_fish_x = x - x_ref
    to_fish_y = y - y_ref

    # Tangent vector (normalized direction)
    tang_len = np.sqrt(dx_dt**2 + dy_dt**2)
    if tang_len < 1e-6:
        tang_len = 1e-6
    tang_x = dx_dt / tang_len
    tang_y = dy_dt / tang_len

    # Cross product gives signed distance (positive = left of path)
    e_ct = tang_x * to_fish_y - tang_y * to_fish_x

    return y_ref, psi_ref, e_ct

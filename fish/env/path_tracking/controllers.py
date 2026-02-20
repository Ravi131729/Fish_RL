import numpy as np
from dynamics import wrap_to_pi


class StanleyController:
    """
    Stanley Controller for path tracking.

    The Stanley method computes steering based on:
    1. Heading error: difference between vehicle heading and path tangent
    2. Cross-track error: lateral distance from the front axle to the path

    Steering angle: delta = (psi_vehicle - psi_path) + arctan(k * e_ct / v)

    where:
    - psi_path: path tangent angle
    - psi_vehicle: vehicle heading
    - k: gain parameter
    - e_ct: cross-track error (positive = left of path)
    - v: forward velocity
    """

    def __init__(self, k=2.0, k_soft=0.1, max_steer=np.pi/3, max_steer_rate=3):

        self.k = k
        self.k_soft = k_soft
        self.max_steer = max_steer
        self.max_steer_rate = max_steer_rate
        self.prev_delta = 0.0
        self.dt = 0.01  # Default timestep, can be updated

    def set_dt(self, dt):
        """Set the timestep for rate limiting."""
        self.dt = dt

    def reset(self):
        """Reset controller state."""
        self.prev_delta = 0.0

    def compute(self, heading, heading_ref, crosstrack_error, velocity):

        heading_error = wrap_to_pi(heading - heading_ref)

        # Cross-track error correction
        v_term = max(abs(velocity), self.k_soft)
        crosstrack_term = np.arctan2(self.k * crosstrack_error, v_term)

        # Stanley control law:
        # Positive delta turns fish right (corrects both heading-left and position-left errors)
        delta = heading_error + crosstrack_term

        # Saturate steering angle
        delta = np.clip(delta, -self.max_steer, self.max_steer)

        # Apply rate limit
        max_delta_change = self.max_steer_rate * self.dt
        delta_change = delta - self.prev_delta
        delta_change = np.clip(delta_change, -max_delta_change, max_delta_change)
        delta = self.prev_delta + delta_change


        self.prev_delta = delta

        return delta


class VelocityController:
    """
    PD Controller for velocity tracking by adjusting amplitude A only.

    Controls the fish's forward velocity by adjusting the propulsion
    amplitude A while keeping frequency w fixed.

    Includes rate limiting to prevent rapid A changes.
    """

    def __init__(self, Kp=100.0, Kd=20.0, A_min=6.0, A_max=13.0,
                 max_A_rate=500.0, dt=0.01):
        """
        Initialize velocity controller.

        Args:
            Kp: Proportional gain
            Kd: Derivative gain
            A_min: Minimum amplitude
            A_max: Maximum amplitude
            max_A_rate: Maximum rate of A change per second
            dt: Timestep for rate limiting
        """
        self.Kp = Kp
        self.Kd = Kd
        self.A_min = A_min
        self.A_max = A_max
        self.max_A_rate = max_A_rate
        self.dt = dt

        self.prev_error = 0.0
        self.prev_A = None  # Will be initialized on first call

    def set_dt(self, dt):
        """Set the timestep for rate limiting."""
        self.dt = dt

    def reset(self):
        """Reset controller state."""
        self.prev_error = 0.0
        self.prev_A = None

    def compute(self, velocity, velocity_ref, w_fixed=3.0):
        """
        Compute amplitude A to track desired velocity (w is fixed).

        Args:
            velocity: Current forward velocity (m/s)
            velocity_ref: Desired forward velocity (m/s)
            w_fixed: Fixed frequency (Hz)

        Returns:
            (A, w): Propulsion amplitude and fixed frequency
        """
        # Velocity error
        error = velocity_ref - velocity

        # Derivative of error
        error_dot = (error - self.prev_error) / self.dt

        # PD control law for A adjustment
        A_cmd = self.Kp * error + self.Kd * error_dot

        # Initialize prev_A on first call
        if self.prev_A is None:
            # Start with mid-range A
            self.prev_A = (self.A_min + self.A_max) / 2

        # Add to previous A (incremental control)
        A = self.prev_A + A_cmd

        # Saturate A
        A = np.clip(A, self.A_min, self.A_max)

        # Apply rate limit for smooth changes
        max_A_change = self.max_A_rate * self.dt
        A_change = A - self.prev_A
        A_change = np.clip(A_change, -max_A_change, max_A_change)
        A = self.prev_A + A_change

        # Store for next iteration
        self.prev_error = error
        self.prev_A = A

        return A, w_fixed


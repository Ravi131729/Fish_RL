"""
Running observation normalizer using Welford's online algorithm.

Usage:
    normalizer = RunningMeanStd(obs_dim)
    normalizer.update(obs_batch)         # (batch, obs_dim) — call outside jit
    normalized = normalizer.normalize(obs)  # works inside or outside jit
"""
import numpy as np
import jax.numpy as jnp
import json, os


class RunningMeanStd:
    """Tracks running mean and variance of observations (Welford's algorithm).

    Stats are stored as numpy arrays so they can be mutated in-place;
    normalize() returns a JAX array for seamless use in jitted code.
    """

    def __init__(self, shape: int, clip: float = 10.0, epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # small init to avoid div-by-zero
        self.clip = clip
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # Stats update (call with each rollout batch, OUTSIDE jit)
    # ------------------------------------------------------------------
    def update(self, x: np.ndarray):
        """Update running stats with a batch of observations.

        Args:
            x: array of shape (..., obs_dim).  Will be reshaped to (M, obs_dim).
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.mean.shape[0])
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Parallel Welford update."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    # ------------------------------------------------------------------
    # Normalize (works inside jit via JAX arrays)
    # ------------------------------------------------------------------
    def normalize(self, obs):
        """Normalize observations: (obs - mean) / std, clipped."""
        mean = jnp.array(self.mean, dtype=jnp.float32)
        std = jnp.sqrt(jnp.array(self.var, dtype=jnp.float32) + self.epsilon)
        return jnp.clip((obs - mean) / std, -self.clip, self.clip)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save normalizer state to a .npz file."""
        np.savez(
            path,
            mean=self.mean,
            var=self.var,
            count=np.array(self.count),
        )

    def load(self, path: str):
        """Load normalizer state from a .npz file."""
        data = np.load(path)
        self.mean = data["mean"]
        self.var = data["var"]
        self.count = float(data["count"])

    def __repr__(self):
        return (
            f"RunningMeanStd(shape={self.mean.shape}, count={self.count:.0f}, "
            f"mean_range=[{self.mean.min():.3f}, {self.mean.max():.3f}], "
            f"std_range=[{np.sqrt(self.var).min():.3f}, {np.sqrt(self.var).max():.3f}])"
        )

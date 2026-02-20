import jax
import jax.numpy as jnp
from flax import nnx

class Actor(nnx.Module):
  def __init__(self,obs_dim,act_dim,*,rngs):
    self.fc1 = nnx.Linear(obs_dim,64,rngs=rngs)
    self.fc2 = nnx.Linear(64,64,rngs=rngs)
    self.mean = nnx.Linear(64,act_dim,rngs=rngs)
    # Use fixed (state-independent) log_std - more stable for PPO
    self.log_std = nnx.Param(jnp.zeros(act_dim))  # init to std=1.0

  def __call__(self,obs):
    x = nnx.relu(self.fc1(obs))
    x = nnx.relu(self.fc2(x))
    mean = self.mean(x)
    # Broadcast log_std to match batch dimension
    log_std = jnp.clip(self.log_std.value, -20, -0.5)  # std in [1.0, ~0.0]
    log_std = jnp.broadcast_to(log_std, mean.shape)
    return mean , log_std

class ValueNet(nnx.Module):
  def __init__(self,obs_dim,*,rngs):
    self.fc1 = nnx.Linear(obs_dim,64,rngs=rngs)
    self.fc2 = nnx.Linear(64,64,rngs=rngs)
    self.v = nnx.Linear(64,1,rngs=rngs)
  def __call__(self,obs):
    x = nnx.relu(self.fc1(obs))
    x = nnx.relu(self.fc2(x))
    return jnp.squeeze(self.v(x), -1)

def sample_action(rng, mean, log_std):
    """Sample action from tanh-squashed Gaussian and return log probability."""
    std = jnp.exp(log_std)
    eps = jax.random.normal(rng, mean.shape)
    pre_tanh = mean + eps * std
    u = jnp.tanh(pre_tanh)

    # Gaussian log prob
    log_prob_gaussian = -0.5 * (jnp.square(eps) + 2 * log_std + jnp.log(2 * jnp.pi)).sum(axis=-1)
    # Jacobian correction for tanh squashing
    log_prob = log_prob_gaussian - jnp.log(1 - u**2 + 1e-6).sum(axis=-1)

    return u, log_prob


def compute_log_prob(mean, log_std, action):

    std = jnp.exp(log_std)


    action_clipped = jnp.clip(action, -1.0 + 1e-6, 1.0 - 1e-6)

    pre_tanh = 0.5 * jnp.log((1 + action_clipped) / (1 - action_clipped))

    # Gaussian log prob of pre_tanh value (same formula as sample_action)
    eps = (pre_tanh - mean) / std
    log_prob_gaussian = -0.5 * (jnp.square(eps) + 2 * log_std + jnp.log(2 * jnp.pi)).sum(axis=-1)


    log_prob = log_prob_gaussian - jnp.log(1 - action_clipped**2 + 1e-6).sum(axis=-1)

    return log_prob
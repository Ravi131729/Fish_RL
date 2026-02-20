import os
import jax
import orbax.checkpoint as ocp
from flax import nnx
from networks import Actor, ValueNet
import optax
from obs_normalizer import RunningMeanStd
from ppo_agent import PPO


def load_agent(checkpoint_name: str, obs_dim: int, act_dim: int):
    """
    Load ONLY actor network from PPO checkpoint.
    Returns:
        actor
        obs_normalizer
    """

    checkpoint_dir = os.path.abspath("./checkpoints")
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # -----------------------------
    # create fresh PPO to get structure
    # -----------------------------
    rng = jax.random.PRNGKey(0)
    agent = PPO(obs_dim, act_dim, rng)
    actor = Actor(obs_dim, act_dim, rngs=nnx.Rngs(rng))

    graphdef, state_template = nnx.split(agent)
    # graphdef, state_template = nnx.split(actor)

    # -----------------------------
    # restore checkpoint state
    # -----------------------------
    checkpointer = ocp.StandardCheckpointer()
    restored_state = checkpointer.restore(ckpt_path, state_template)
    checkpointer.wait_until_finished()

    # merge graphdef + restored state
    agent = nnx.merge(graphdef, restored_state)
    # actor = nnx.merge(graphdef, restored_state)
    print(f"Loaded checkpoint from {ckpt_path}")

    # -----------------------------
    # extract ONLY actor
    # -----------------------------
    actor = agent.actor

    # -----------------------------
    # load obs normalizer
    # -----------------------------
    normalizer = RunningMeanStd(obs_dim)

    norm_path = ckpt_path + "_obs_normalizer.npz"
    if not os.path.exists(norm_path):
        norm_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_obs_normalizer.npz")

    if os.path.exists(norm_path):
        normalizer.load(norm_path)
        print(f"Loaded obs normalizer from {norm_path}")
        print(f"  {normalizer}")
    else:
        print("WARNING: No obs normalizer found → identity")

    return agent, normalizer

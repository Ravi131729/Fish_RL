import os
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
import wandb

# --------- import YOUR env ----------
# Change this import to wherever you put the code you pasted.
from fish_env import EnvConfig, EnvState, reset_env, step_env, build_obs, compute_reward


# ----------------- config -----------------
@dataclass
class TrainConfig:
    # experiment
    project: str = "diff-bptt-fish"
    run_name: str = "bptt_policy"
    seed: int = 0
    out_dir: str = "./checkpoints_diff"

    # env
    n_envs: int = 256
    nx: int = 7                 # state dim you used in reset_env (x is (N, nx))
    horizon: int = 256          # rollout length for BPTT
    eval_horizon: int = 512     # eval rollout length
    eval_every: int = 25
    reset_every: int = 1        # reset envs every k updates (helps avoid very long BPTT)

    # optimization
    total_updates: int = 1000
    lr: float = 3e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    entropy_bonus: float = 0.0
    action_smooth_penalty: float = 0.0  # optional: penalize delta_raw changes

    # checkpointing
    save_every: int = 50
    keep_last: int = 3
    keep_best: int = 1          # you can keep more if you want

    # misc
    use_wandb: bool = True
    wandb_mode: str = "online"  # "offline" also works
    precision: str = "float32"  # "float32" or "float64"


# ----------------- policy -----------------
class Policy(nnx.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(obs_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.out = nnx.Linear(hidden, act_dim, rngs=rngs)

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        # obs: (N, obs_dim)
        x = nnx.relu(self.fc1(obs))
        x = nnx.relu(self.fc2(x))
        a = jnp.tanh(self.out(x))  # bounded in [-1,1]
        return a


# ----------------- utilities -----------------
def tree_l2_norm(tree):
    leaves = jax.tree.leaves(tree)
    return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in leaves if x is not None]))


def make_optimizer(cfg: TrainConfig):

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay),
    )
    return tx


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ----------------- rollout / loss -----------------
def rollout_loss_fn(
    policy: Policy,
    init_state: EnvState,
    key: jax.Array,
    env_cfg: EnvConfig,
    horizon: int,
    action_smooth_penalty: float,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:


    def scan_step(carry, _):
        state, key, prev_a = carry
        key, sub = jax.random.split(key)

        obs = build_obs(state)          # (N, obs_dim)
        a = policy(obs)                # (N, act_dim=1) in [-1,1]
        a = a.squeeze(-1)              # (N,)

        next_state = step_env(state, a, sub, env_cfg)  # differentiable

        r = compute_reward(state, next_state, a, env_cfg)  # (N,)

        # optional smoothness regularizer on delta_raw changes
        smooth = jnp.square(a - prev_a)  # (N,)

        carry2 = (next_state, key, a)
        out = (r, smooth)
        return carry2, out


    prev_a0 = jnp.zeros((init_state.x.shape[0],), dtype=init_state.x.dtype)

    (final_state, _key, _), (rews, smooths) = jax.lax.scan(
        scan_step,
        (init_state, key, prev_a0),
        None,
        length=horizon,
    )


    ep_return = jnp.sum(rews, axis=0)        # (N,)
    mean_return = jnp.mean(ep_return)

    smooth_cost = jnp.mean(jnp.sum(smooths, axis=0))  # scalar (sum over time, mean over envs)

    # We maximize return -> minimize negative return
    loss = -mean_return + action_smooth_penalty * smooth_cost

    metrics = {
        "loss": loss,
        "mean_return": mean_return,
        "smooth_cost": smooth_cost,
        "final_time_mean": jnp.mean(final_state.t),
    }
    return loss, metrics


# jit value_and_grad wrapper
@jax.jit
def loss_and_grad(
    policy: Policy,
    init_state: EnvState,
    key: jax.Array,
    env_cfg: EnvConfig,
    horizon: int,
    action_smooth_penalty: float,
):
    (loss, metrics), grads = jax.value_and_grad(
        lambda p: rollout_loss_fn(p, init_state, key, env_cfg, horizon, action_smooth_penalty),
        has_aux=True,
    )(policy)
    return loss, metrics, grads


@jax.jit
def apply_updates(policy: Policy, opt_state, grads, tx):
    updates, opt_state = tx.update(grads, opt_state, params=policy)
    policy = nnx.apply_updates(policy, updates)
    return policy, opt_state


# ----------------- eval -----------------
@jax.jit
def eval_rollout(policy: Policy, init_state: EnvState, key: jax.Array, env_cfg: EnvConfig, horizon: int):
    def scan_step(carry, _):
        state, key = carry
        key, sub = jax.random.split(key)
        obs = build_obs(state)
        a = policy(obs).squeeze(-1)
        next_state = step_env(state, a, sub, env_cfg)
        r = compute_reward(state, next_state, a, env_cfg)
        return (next_state, key), r

    (final_state, _), rews = jax.lax.scan(scan_step, (init_state, key), None, length=horizon)
    ep_return = jnp.sum(rews, axis=0)
    return {
        "eval_mean_return": jnp.mean(ep_return),
        "eval_min_return": jnp.min(ep_return),
        "eval_max_return": jnp.max(ep_return),
        "eval_final_time_mean": jnp.mean(final_state.t),
    }


# ----------------- checkpointing -----------------
class Checkpointer:
    def __init__(self, out_dir: str, keep_last: int = 3):
        ensure_dir(out_dir)
        self.out_dir = out_dir
        self.keep_last = keep_last
        self.checkpointer = ocp.PyTreeCheckpointer()

        self.options = ocp.CheckpointManagerOptions(
            max_to_keep=keep_last,
            create=True,
        )
        self.mngr = ocp.CheckpointManager(out_dir, self.checkpointer, options=self.options)

    def save(self, step: int, payload: Dict[str, Any]):
        self.mngr.save(step, payload)
        self.mngr.wait_until_finished()

    def latest_step(self) -> Optional[int]:
        return self.mngr.latest_step()

    def restore_latest(self) -> Optional[Dict[str, Any]]:
        s = self.latest_step()
        if s is None:
            return None
        return self.mngr.restore(s)

    def close(self):
        self.mngr.close()


def save_best(best_dir: str, payload: Dict[str, Any]):
    ensure_dir(best_dir)
    ckpt = ocp.PyTreeCheckpointer()
    ckpt.save(best_dir, payload, force=True)


def restore_best(best_dir: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(best_dir):
        return None
    ckpt = ocp.PyTreeCheckpointer()
    try:
        return ckpt.restore(best_dir)
    except Exception:
        return None


# ----------------- main training -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_updates", type=int, default=None)
    parser.add_argument("--n_envs", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.total_updates is not None:
        cfg.total_updates = args.total_updates
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    if args.horizon is not None:
        cfg.horizon = args.horizon

    # dtype
    if cfg.precision == "float64":
        jax.config.update("jax_enable_x64", True)

    # build your EnvConfig here (fill with your actual numbers)
    # NOTE: Put your real values; below are placeholders.
    env_cfg = EnvConfig(
        dt=0.02,
        max_steps=2000,
        A_min=0.5,
        A_max=2.0,
        w_min=1.0,
        w_max=6.0,
        I=1.0,
        delta_max=0.5,
        delta_rate_max=2.0,
        max_position=5.0,
    )

    # wandb
    if cfg.use_wandb:
        wandb.init(
            project=cfg.project,
            name=cfg.run_name,
            config=asdict(cfg) | {"env_cfg": {k: getattr(env_cfg, k) for k in env_cfg.__dict__.keys()}},
            mode=cfg.wandb_mode,
        )

    # rngs
    key = jax.random.PRNGKey(cfg.seed)
    key, k_model, k_reset = jax.random.split(key, 3)

    # one dummy reset to infer obs_dim
    init_state = reset_env(k_reset, cfg.n_envs, cfg.nx, env_cfg)
    obs_dim = int(build_obs(init_state).shape[1])
    act_dim = 1

    # policy + optimizer
    policy = Policy(obs_dim, act_dim, hidden=64, rngs=nnx.Rngs(k_model))
    tx = make_optimizer(cfg)
    opt_state = tx.init(policy)

    # checkpoint managers
    ensure_dir(cfg.out_dir)
    mgr = Checkpointer(cfg.out_dir, keep_last=cfg.keep_last)
    best_dir = os.path.join(cfg.out_dir, "best")

    start_update = 0
    best_eval = -jnp.inf

    # optionally resume
    if args.resume:
        restored = mgr.restore_latest()
        if restored is not None:
            policy = restored["policy"]
            opt_state = restored["opt_state"]
            start_update = int(restored["update"] + 1)
            best_eval = restored.get("best_eval", best_eval)
            key = restored.get("key", key)
            print(f"[resume] from update={start_update} best_eval={float(best_eval):.3f}")

        best_restored = restore_best(best_dir)
        if best_restored is not None and "best_eval" in best_restored:
            best_eval = jnp.maximum(best_eval, best_restored["best_eval"])

    # training loop
    wall0 = time.time()
    state = init_state

    for update in range(start_update, cfg.total_updates):
        # periodic reset to keep BPTT stable (recommended)
        if (update % cfg.reset_every) == 0:
            key, k_reset = jax.random.split(key)
            state = reset_env(k_reset, cfg.n_envs, cfg.nx, env_cfg)

        key, k_roll = jax.random.split(key)

        t0 = time.time()
        loss, metrics, grads = loss_and_grad(
            policy,
            state,
            k_roll,
            env_cfg,
            cfg.horizon,
            cfg.action_smooth_penalty,
        )

        grad_norm = tree_l2_norm(grads)

        policy, opt_state = apply_updates(policy, opt_state, grads, tx)
        dt_train = time.time() - t0

        # log train
        log_dict = {
            "update": update,
            "time/iter_sec": dt_train,
            "time/wall_sec": time.time() - wall0,
            "train/loss": float(metrics["loss"]),
            "train/mean_return": float(metrics["mean_return"]),
            "train/smooth_cost": float(metrics["smooth_cost"]),
            "train/grad_norm": float(grad_norm),
            "train/lr": cfg.lr,
        }

        # eval
        if (update % cfg.eval_every) == 0:
            key, k_eval_reset, k_eval = jax.random.split(key, 3)
            eval_state = reset_env(k_eval_reset, cfg.n_envs, cfg.nx, env_cfg)
            eval_metrics = eval_rollout(policy, eval_state, k_eval, env_cfg, cfg.eval_horizon)

            log_dict |= {f"eval/{k}": float(v) for k, v in eval_metrics.items()}

            eval_score = float(eval_metrics["eval_mean_return"])
            if eval_score > float(best_eval):
                best_eval = jnp.array(eval_score)
                # save a "best" checkpoint (single directory)
                save_best(best_dir, {
                    "policy": policy,
                    "best_eval": best_eval,
                    "update": update,
                })
                log_dict["checkpoint/new_best"] = 1.0
            else:
                log_dict["checkpoint/new_best"] = 0.0

        # periodic checkpoint
        if (update % cfg.save_every) == 0:
            mgr.save(update, {
                "policy": policy,
                "opt_state": opt_state,
                "update": update,
                "best_eval": best_eval,
                "key": key,
                "cfg": asdict(cfg),
            })
            log_dict["checkpoint/saved"] = 1.0
        else:
            log_dict["checkpoint/saved"] = 0.0

        # wandb / print
        if cfg.use_wandb:
            wandb.log(log_dict, step=update)

        if (update % 10) == 0:
            msg = f"upd {update:5d} | loss {log_dict['train/loss']:+.3f} | ret {log_dict['train/mean_return']:+.3f} | g {log_dict['train/grad_norm']:.3f}"
            if f"eval/eval_mean_return" in log_dict:
                msg += f" | eval {log_dict['eval/eval_mean_return']:+.3f} (best {float(best_eval):+.3f})"
            msg += f" | {dt_train:.3f}s"
            print(msg, flush=True)

    # final save
    mgr.save(cfg.total_updates, {
        "policy": policy,
        "opt_state": opt_state,
        "update": cfg.total_updates,
        "best_eval": best_eval,
        "key": key,
        "cfg": asdict(cfg),
    })
    mgr.close()

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

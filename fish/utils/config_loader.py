import yaml
from fish.env.types import EnvConfig
from fish.agents.ppo_agent import PPOConfig

def load_config(path: str):

  with open(path, "r") as f:
      cfg_dict = yaml.safe_load(f)

      env_cfg = EnvConfig(**cfg_dict["env"])


      eval_dict = cfg_dict.get("eval", {})
      eval_cfg = env_cfg.replace(**eval_dict)

      ppo_cfg = PPOConfig(**cfg_dict["ppo"])

      training_cfg = cfg_dict["training"]
      logging_cfg = cfg_dict.get("logging", {})
      action_cfg = cfg_dict.get("action", {"dim": 1})

      seed = cfg_dict.get("seed", 0)

  return env_cfg, eval_cfg, ppo_cfg, training_cfg, logging_cfg, action_cfg, seed


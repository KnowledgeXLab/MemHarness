# Copyright 2025 the MemAdaptor team.
"""Infer train/eval game counts from AlfWorld (AlfredTWEnv) without Ray."""
from __future__ import annotations

import os


def infer_alfworld_num_games(
    config_path: str | None = None,
    eval_split: str = "eval_in_distribution",
) -> tuple[int, int]:
    """
    Load AlfredTWEnv twice (train + eval split) and return (num_train_games, num_eval_games)
    after AlfWorld filtering (solvable, task_types, etc.).

    Args:
        config_path: Path to config_tw.yaml. Default: agent_system/.../config_tw.yaml under repo root.
        eval_split: 'eval_in_distribution' or 'eval_out_of_distribution' (matches env_manager / alfred_tw_env).

    Returns:
        (n_train, n_eval)
    """
    if config_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.normpath(os.path.join(here, "..", ".."))
        config_path = os.path.join(
            repo_root,
            "agent_system",
            "environments",
            "env_package",
            "alfworld",
            "configs",
            "config_tw.yaml",
        )
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"AlfWorld config not found: {config_path}")

    import yaml

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Local package path (vendored alfworld under agent_system)
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import (
        AlfredTWEnv,
    )

    train_env = AlfredTWEnv(config, train_eval="train")
    n_train = int(train_env.num_games)
    del train_env

    if eval_split not in ("eval_in_distribution", "eval_out_of_distribution"):
        raise ValueError(f"Unknown eval_split: {eval_split}")
    val_env = AlfredTWEnv(config, train_eval=eval_split)
    n_eval = int(val_env.num_games)
    del val_env

    return n_train, n_eval

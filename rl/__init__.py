# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the Spider RESIDUAL-gait rsl_rl tasks.

Tasks:
  Isaac-Velocity-Flat-Spider-Residual-v0        (+ -Play-v0)
  Isaac-Velocity-Rough-Spider-Residual-v0       (+ -Play-v0)
"""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Velocity-Flat-Spider-Residual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:SpiderResidualFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpiderResidualFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Spider-Residual-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:SpiderResidualFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpiderResidualFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Spider-Residual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:SpiderResidualRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpiderResidualRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Spider-Residual-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:SpiderResidualRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpiderResidualRoughPPORunnerCfg",
    },
)

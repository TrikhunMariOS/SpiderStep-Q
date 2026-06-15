# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP terms for the Spider residual-gait RL task.

Re-exports Isaac Lab's standard velocity-task mdp terms plus our custom action,
observation and reward terms, so env configs can `from . import mdp` and use both.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

# custom terms
from .gait_residual_action import GaitResidualAction, GaitResidualActionCfg  # noqa: F401
from .gait_foot_offset_action import GaitFootOffsetAction, GaitFootOffsetActionCfg  # noqa: F401
from .observations import gait_phase  # noqa: F401
from .rewards import (  # noqa: F401
    foot_swing_suppression,
    foot_clearance_over_terrain,
    foot_touchdown_impact,
)

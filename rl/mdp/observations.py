# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation terms for the residual gait policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, action_term_name: str = "joint_pos") -> torch.Tensor:
    """Global gait phase as (sin, cos) so the policy knows where in the step cycle it is.

    Read straight from the GaitResidualAction term so there's no separate clock to drift.
    Returns [num_envs, 2] = (sin(2*pi*phase), cos(2*pi*phase)).
    """
    term = env.action_manager._terms[action_term_name]
    return term.gait_phase_obs

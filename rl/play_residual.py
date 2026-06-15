# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Launcher: register the Spider residual-gait tasks, then run Isaac Lab's rsl_rl play.py.

Same idea as train_residual.py — see that file for the full explanation.

Usage:
    isaaclab.bat -p scripts/MY_Final/rl/play_residual.py ^
        --task Isaac-Velocity-Flat-Spider-Residual-Play-v0 --num_envs 16 ^
        --checkpoint logs/rsl_rl/spider_residual_flat/<run>/model_<n>.pt
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
_SCRIPTS = os.path.dirname(_PROJ)
_RSL = os.path.join(_SCRIPTS, "reinforcement_learning", "rsl_rl")

# 1) register our tasks (lazy, no Isaac app yet)
sys.path.insert(0, _PROJ)
import rl  # noqa: F401,E402

# 2) let Isaac's play.py find its sibling module `cli_args`
sys.path.insert(0, _RSL)

# 3) hand off to the real play.py
_real_play = os.path.join(_RSL, "play.py")
sys.argv = [_real_play] + sys.argv[1:]
runpy.run_path(_real_play, run_name="__main__")

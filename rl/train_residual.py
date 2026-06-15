# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Launcher: register our gym tasks, then run Isaac Lab's rsl_rl train.py.

Isaac Lab's train.py only auto-discovers tasks inside isaaclab_tasks, so we import our
`rl` package first (lazy gym.register, no app needed) then hand off to the real train.py.

Usage (same flags as the normal train.py):
    isaaclab.bat -p scripts/MY_Final/rl/train_residual.py ^
        --task Isaac-Velocity-Flat-Spider-Residual-v0 --num_envs 1024 --headless
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # .../scripts/MY_Final/rl
_PROJ = os.path.dirname(_HERE)                           # .../scripts/MY_Final
_SCRIPTS = os.path.dirname(_PROJ)                        # .../scripts
_RSL = os.path.join(_SCRIPTS, "reinforcement_learning", "rsl_rl")

# 1) make `import rl` work, then register the gym tasks (lazy -> no Isaac yet)
sys.path.insert(0, _PROJ)
import rl  # noqa: F401,E402   -> runs gym.register(Isaac-Velocity-*-Spider-Residual-*)

# 2) let Isaac's train.py find its sibling module `cli_args`
sys.path.insert(0, _RSL)

# 3) hand off to the real train.py (launches the app, finds our registered task)
_real_train = os.path.join(_RSL, "train.py")
sys.argv = [_real_train] + sys.argv[1:]
runpy.run_path(_real_train, run_name="__main__")

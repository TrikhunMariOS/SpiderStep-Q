# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Validates that the batched torch engine matches the numpy reference (test5 math).

If this fails, don't proceed to RL — the baseline would be wrong. Checks pure IK, the
full gait pipeline, and float32 sanity (the training dtype). Tolerances: float64 < 1e-6
(must pass), float32 < 2e-3 (informational).

Run:  python test_ik_validation.py
"""

import sys

import numpy as np
import torch

import gait_numpy_ref as ref
from gait_torch import SpiderGaitEngine, JOINT_NAMES_FLAT

# Windows Thai console (cp874) cannot encode em-dashes / arrows; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

torch.manual_seed(0)
np.random.seed(0)

TOL64 = 1e-6      # hard pass/fail for float64
TOL32 = 2e-3      # informational ceiling for float32


# --------------------------------------------------------------------------- #
#  helpers                                                                    #
# --------------------------------------------------------------------------- #
def _report(name, err, tol, unit, hard=True):
    ok = err < tol
    flag = "PASS" if ok else "FAIL"
    gate = "" if hard else "  (informational)"
    print(f"  [{flag}] {name:<34} max err = {err:.3e} {unit}  (tol {tol:.0e}){gate}")
    return ok or (not hard)


# --------------------------------------------------------------------------- #
#  Test 1 — pure IK on random foot positions                                  #
# --------------------------------------------------------------------------- #
def test_pure_ik(n=4000):
    print("\n[Test 1] Pure IK  —  random foot positions -> joint angles")

    # sample foot targets in body frame (mm) around each leg's nominal stance,
    # wide enough to cover swing height and to occasionally trigger the reach clamp
    nominal = np.array([ref.GAIT_FOOT_BODY[l] for l in ref.LEGS])      # [4,3]
    lo = nominal + np.array([-90.0, -90.0, -45.0])
    hi = nominal + np.array([ 90.0,  90.0,  95.0])
    foot = np.random.uniform(lo, hi, size=(n, 4, 3))                  # [n,4,3]

    # numpy reference
    q_ref = np.stack([ref.numpy_ik_body_array(foot[i]) for i in range(n)])  # [n,4,3]

    # torch engine (float64)
    eng = SpiderGaitEngine(num_envs=n, device="cpu", dtype=torch.float64)
    q_torch = eng.ik(torch.tensor(foot, dtype=torch.float64)).cpu().numpy()  # [n,4,3]

    err = np.abs(q_ref - q_torch).max()
    return _report("IK joint angle (float64)", err, TOL64, "rad")


# --------------------------------------------------------------------------- #
#  Test 2 — full gait pipeline                                                #
# --------------------------------------------------------------------------- #
def _sample_commands(n):
    """Mix of: forward walk, sideways, turning, stand-still, and over-max stride."""
    cmd = np.zeros((n, 3))
    cmd[:, 0] = np.random.uniform(-300.0, 300.0, n)   # vx mm/s
    cmd[:, 1] = np.random.uniform(-200.0, 200.0, n)   # vy mm/s
    cmd[:, 2] = np.random.uniform(-60.0, 60.0, n)     # wz deg/s
    # force some edge cases
    cmd[: n // 10] = 0.0                               # stand still
    cmd[n // 10 : n // 5, 0] = 600.0                   # huge vx -> stride clamp
    cmd[n // 5 : n // 4, 2] = 0.0                      # pure translation
    return cmd


def test_full_pipeline(n=4000):
    print("\n[Test 2] Full pipeline  —  (gait time, command) -> joint angles")

    t = np.random.uniform(0.0, 5.0, n)
    t[: n // 20] = 0.0          # t=0 : CoG ramp just starting
    t[n // 20 : n // 10] = np.random.uniform(0.0, 0.6, n // 20)  # inside ramp
    cmd = _sample_commands(n)

    # numpy reference (foot targets + joint targets)
    foot_ref = np.stack([ref.numpy_foot_targets(float(t[i]), tuple(cmd[i])) for i in range(n)])
    q_ref    = np.stack([ref.numpy_ik_body_array(foot_ref[i]) for i in range(n)])

    # torch engine (float64)
    eng = SpiderGaitEngine(num_envs=n, device="cpu", dtype=torch.float64)
    tt  = torch.tensor(t, dtype=torch.float64)
    cc  = torch.tensor(cmd, dtype=torch.float64)
    foot_torch = eng.foot_targets_with_cog(tt, cc).cpu().numpy()
    q_torch    = eng.joint_targets(tt, cc, flat=False).cpu().numpy()

    err_foot = np.abs(foot_ref - foot_torch).max()
    err_q    = np.abs(q_ref - q_torch).max()
    ok1 = _report("foot target after CoG (float64)", err_foot, 1e-4, "mm")
    ok2 = _report("joint angle full pipe (float64)", err_q, TOL64, "rad")
    return ok1 and ok2


# --------------------------------------------------------------------------- #
#  Test 3 — float32 sanity (the dtype used during training)                   #
# --------------------------------------------------------------------------- #
def test_float32(n=4000):
    print("\n[Test 3] float32 sanity  —  training dtype error vs numpy")

    t = np.random.uniform(0.0, 5.0, n)
    cmd = _sample_commands(n)
    q_ref = np.stack([ref.numpy_joint_targets(float(t[i]), tuple(cmd[i])) for i in range(n)])

    eng = SpiderGaitEngine(num_envs=n, device="cpu", dtype=torch.float32)
    q32 = eng.joint_targets(
        torch.tensor(t, dtype=torch.float32),
        torch.tensor(cmd, dtype=torch.float32),
        flat=False,
    ).cpu().numpy()

    err = np.abs(q_ref - q32).max()
    return _report("joint angle (float32)", err, TOL32, "rad", hard=False)


# --------------------------------------------------------------------------- #
#  Test 4 — batched determinism / shape sanity                                #
# --------------------------------------------------------------------------- #
def test_shapes():
    print("\n[Test 4] Shapes & ordering")
    eng = SpiderGaitEngine(num_envs=8, device="cpu", dtype=torch.float32)
    t = torch.zeros(8)
    cmd = torch.zeros(8, 3); cmd[:, 0] = 80.0
    flat = eng.joint_targets(t, cmd, flat=True)
    grid = eng.joint_targets(t, cmd, flat=False)
    ok = (flat.shape == (8, 12)) and (grid.shape == (8, 4, 3))
    ok = ok and len(JOINT_NAMES_FLAT) == 12
    print(f"  [{'PASS' if ok else 'FAIL'}] flat {tuple(flat.shape)}  grid {tuple(grid.shape)}")
    print(f"        joint order: {JOINT_NAMES_FLAT}")
    return ok


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 72)
    print("Phase 0 validation  —  gait_torch  vs  gait_numpy_ref (test5 math)")
    print("=" * 72)

    results = [
        test_pure_ik(),
        test_full_pipeline(),
        test_float32(),
        test_shapes(),
    ]

    print("\n" + "=" * 72)
    if all(results):
        print("RESULT:  ALL PASS  [OK]   torch port matches test5.  Safe to wire into RL.")
    else:
        print("RESULT:  FAILED   [X]    fix gait_torch before proceeding to the RL env.")
    print("=" * 72)
    raise SystemExit(0 if all(results) else 1)

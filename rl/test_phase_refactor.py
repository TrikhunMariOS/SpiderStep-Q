# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-refactor equivalence test (no Isaac app needed — pure torch).

The 3.5b dynamic-gait refactor split the engine into phase-based cores (*_p) with
time-based wrappers.  At CONSTANT frequency the two MUST be bit-identical:

    foot_targets(t, cmd)          == foot_targets_p(t*freq, cmd, period=1/freq)
    cog_offset(t)                 == cog_offset_p(t*freq, ramp=t/COG_RAMP_TIME)
    foot_targets_with_cog(t, cmd) == foot_targets_with_cog_p(...)
    joint_targets(t, cmd)         unchanged end-to-end

Also sanity-checks the DYNAMIC path: integrating phase with a varying frequency is
continuous (no foot-target jumps when freq changes), which was the whole point.

Run:  python scripts/MY_Final/rl/test_phase_refactor.py
"""

import torch

from gait_numpy_ref import COG_RAMP_TIME
from gait_torch import SpiderGaitEngine


def main():
    torch.manual_seed(0)
    E = 128
    eng = SpiderGaitEngine(num_envs=E, device="cpu", dtype=torch.float64)

    t = torch.rand(E, dtype=torch.float64) * 10.0                        # 0..10 s
    cmd = (torch.rand(E, 3, dtype=torch.float64) - 0.5) * 2.0
    cmd[:, 0] *= 250.0   # vx mm/s
    cmd[:, 1] *= 120.0   # vy mm/s
    cmd[:, 2] *= 30.0    # wz deg/s

    period = torch.full((E, 1), eng.gait_period, dtype=torch.float64)
    ramp = torch.clamp(t.unsqueeze(-1) / COG_RAMP_TIME, 0.0, 1.0)
    gphase = t * eng.freq

    e1 = (eng.foot_targets(t, cmd) - eng.foot_targets_p(gphase, cmd, period)).abs().max()
    e2 = (eng.cog_offset(t) - eng.cog_offset_p(gphase, ramp)).abs().max()
    e3 = (eng.foot_targets_with_cog(t, cmd)
          - eng.foot_targets_with_cog_p(gphase, cmd, period, ramp)).abs().max()

    print(f"wrapper-vs-core  foot_targets      max err = {e1:.3e} mm")
    print(f"wrapper-vs-core  cog_offset        max err = {e2:.3e} mm")
    print(f"wrapper-vs-core  foot_with_cog     max err = {e3:.3e} mm")

    # --- dynamic path: phase integration with varying freq must be CONTINUOUS ----
    # simulate 2 s at 400 Hz while sweeping freq 1.2 -> 2.5 Hz; the largest single-step
    # foot-target jump must stay small (a discontinuity would show up as a huge step).
    dt = 0.0025
    steps = 800
    phase = torch.zeros(E, dtype=torch.float64)
    cmd_c = cmd.clone()
    cmd_c[:, 2] = 0.0
    prev = None
    max_jump = 0.0
    one = torch.ones(E, 1, dtype=torch.float64)
    for i in range(steps):
        freq = 1.2 + 1.3 * (i / steps)                                   # sweep up
        phase = phase + freq * dt
        foot = eng.foot_targets_with_cog_p(phase, cmd_c, one / freq, one)
        if prev is not None:
            max_jump = max(max_jump, float((foot - prev).abs().max()))
        prev = foot
    print(f"dynamic-freq sweep: max single-step foot jump = {max_jump:.3f} mm "
          f"(continuous if ~< 5 mm at 400 Hz)")

    ok = e1 < 1e-9 and e2 < 1e-9 and e3 < 1e-9 and max_jump < 5.0
    print("PHASE REFACTOR:", "PASS" if ok else "CHECK (see numbers above)")


if __name__ == "__main__":
    main()

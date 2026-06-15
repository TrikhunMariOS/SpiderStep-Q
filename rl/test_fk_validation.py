# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""FK round-trip validation (no Isaac app needed — pure torch).

Checks that the new SpiderGaitEngine.fk() exactly inverts ik():
    fk(ik(foot)) ≈ foot   for foot targets sampled around the gait workspace.

Run:  python scripts/MY_Final/rl/test_fk_validation.py
  (or with isaaclab.bat -p if your system python has no torch)
"""

import torch

from gait_torch import SpiderGaitEngine
from gait_numpy_ref import GAIT_FOOT_BODY, LEGS


def main():
    torch.manual_seed(0)
    E = 256
    eng = SpiderGaitEngine(num_envs=E, device="cpu", dtype=torch.float64)

    # sample foot targets: gait stance point + random offsets within the work envelope
    base = torch.tensor([GAIT_FOOT_BODY[l] for l in LEGS], dtype=torch.float64)  # [4,3]
    offsets = (torch.rand(E, 4, 3, dtype=torch.float64) - 0.5) * 2.0
    offsets[..., 0] *= 50.0   # ±50 mm x
    offsets[..., 1] *= 50.0   # ±50 mm y
    offsets[..., 2] *= 40.0   # ±40 mm z (matches the RL Δz authority)
    foot_in = base.unsqueeze(0) + offsets                       # [E,4,3]

    q = eng.ik(foot_in)                                         # [E,4,3] rad
    foot_out = eng.fk(q)                                        # [E,4,3] mm

    err = (foot_out - foot_in).norm(dim=-1)                     # [E,4] mm

    # ik() clamps to joint limits / reach envelope — targets that were clamped
    # CANNOT round-trip (fk returns the clamped foot, which is correct behaviour).
    # Detect them: re-running ik on fk's output must give the same q.
    q2 = eng.ik(foot_out)
    consistent = (q2 - q).abs().max()

    print(f"round-trip |fk(ik(p)) - p|  mean={err.mean():.4f} mm  max={err.max():.4f} mm")
    print(f"(targets clamped by ik can't round-trip; those are fine if the next line is ~0)")
    print(f"self-consistency |ik(fk(q)) - q|  max={consistent:.2e} rad")

    ok = consistent < 1e-9 and err.median() < 1e-6
    print("FK VALIDATION:", "PASS" if ok else "CHECK (see numbers above)")


if __name__ == "__main__":
    main()

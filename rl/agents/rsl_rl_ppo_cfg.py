# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""rsl_rl PPO configs for the Spider residual-gait velocity tasks."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class SpiderResidualRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "spider_unified"   # consolidated walker (fixed cadence, trimmed residual)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.35,   # 0.2 was too small to damp the ±13° wobble; entropy_coef keeps it bounded
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,   # 0.01 was too high for a residual task (action std exploded)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class SpiderResidualFlatPPORunnerCfg(SpiderResidualRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        # with domain randomization (push + mass + friction) the task is harder; stop early
        # once bad_orientation stays ~0 and reward plateaus (watch tensorboard)
        self.max_iterations = 1500
        self.save_interval = 50
        self.experiment_name = "spider_speed_flat"   # flat SPEED arena
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]

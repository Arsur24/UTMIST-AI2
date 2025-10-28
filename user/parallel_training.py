"""
Parallel Training Module for GPU-accelerated training
This module contains train_parallel and related utilities
"""

import os
import time
import torch
import numpy as np
from functools import partial
from typing import Optional
import threading
import glob
from stable_baselines3.common import vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

from environment.agent import (
    Agent, SelfPlayWarehouseBrawl, RewardManager,
    SaveHandler, OpponentsCfg, run_match, run_real_time_match,
    BasedAgent, CameraResolution, TrainLogging
)

import pygame


class DemoMatchCallback(BaseCallback):
    """Callback that runs demo matches after checkpoints are saved"""
    def __init__(self, agent, save_handler, demo_interval_steps=10000, verbose=0):
        super().__init__(verbose)
        self.agent = agent
        self.save_handler = save_handler
        self.demo_interval_steps = demo_interval_steps
        self.last_demo_timesteps = 0
        self.demo_count = 0

    def _on_step(self) -> bool:
        """Called after each training step"""
        # Check if it's time for a demo (every demo_interval_steps)
        if self.num_timesteps - self.last_demo_timesteps >= self.demo_interval_steps:
            self.demo_count += 1
            self.last_demo_timesteps = self.num_timesteps

            print(f"\n{'='*70}")
            print(f"🎮 DEMO MATCH #{self.demo_count} - After {self.num_timesteps:,} timesteps")
            print(f"{'='*70}")

            try:
                # Import here to avoid circular imports
                from user.train_agent import BasedAgent

                print(f"Opening pygame window for demo match...")

                # Run REAL-TIME demo match (shows actual pygame window!)
                match_stats = run_real_time_match(
                    agent_1=self.agent,
                    agent_2=BasedAgent(),
                    max_timesteps=2700,  # 90 seconds at 30 FPS
                    resolution=CameraResolution.LOW,
                )

                print(f"✓ Demo Match Result: {match_stats.player1_result}")
                print(f"  Match time: {match_stats.match_time:.1f}s")
                print(f"  Player 1 lives: {match_stats.player1.lives_left}")
                print(f"  Player 2 lives: {match_stats.player2.lives_left}")
                print(f"{'='*70}\n")

            except Exception as e:
                print(f"⚠ Demo match error: {e}")
                import traceback
                traceback.print_exc()

        return True  # Continue training


def train_parallel(
    agent: Agent,
    reward_manager: RewardManager,
    save_handler: Optional[SaveHandler] = None,
    opponent_cfg: OpponentsCfg = OpponentsCfg(),
    resolution: CameraResolution = CameraResolution.LOW,
    train_timesteps: int = 400_000,
    train_logging: TrainLogging = TrainLogging.PLOT,
    n_envs: int = 16,
    train_epochs: Optional[int] = None,
    fast_mode: bool = False,
    demo_interval_steps: int = 10000,  # Run demo match every 10k steps
):
    """
    Train agent with parallel environments using CUDA acceleration.

    Args:
        agent: The agent to train
        reward_manager: Reward manager
        save_handler: Handler for saving checkpoints
        opponent_cfg: Opponent configuration
        resolution: Camera resolution
        train_timesteps: Total timesteps (ignored if train_epochs is set)
        train_logging: Logging mode
        n_envs: Number of parallel environments (default changed to 16 for ~16x speedup target)
        train_epochs: Number of epochs (overrides train_timesteps)
        fast_mode: Fast mode for quick testing
        show_live_training: Show live pygame window during training
    """

    # DON'T start visualizer yet - must create subprocesses first to avoid pickling pygame objects!
    visualizer = None

    # ============================================================
    # 1. Device info
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if save_handler is not None and not hasattr(save_handler, 'num_timesteps'):
        save_handler.num_timesteps = 0
    print(f"\n{'=' * 60}")
    print(f"Training on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Parallel environments: {n_envs} (default set to 16 for target ~16x speedup)")
    print(f"{'=' * 60}\n")

    # ============================================================
    # 2. Epoch -> timestep conversion
    # ============================================================
    if train_epochs is not None:
        if fast_mode:
            train_timesteps = train_epochs * 90
            print(f"FAST MODE: {train_epochs} epochs ≈ {train_timesteps:,} timesteps")
        else:
            train_timesteps = train_epochs * 2700
            print(f"Training for {train_epochs} epochs ≈ {train_timesteps:,} timesteps")

    # ============================================================
    # 3. Environment creation function
    # ============================================================
    def make_env(env_idx=0):
        def _init():
            if env_idx == 0:
                env = SelfPlayWarehouseBrawl(
                    reward_manager=reward_manager,
                    opponent_cfg=opponent_cfg,
                    save_handler=save_handler,
                    resolution=resolution,
                )
            else:
                # Strip self-play for workers
                worker_opponents = {
                    k: v for k, v in opponent_cfg.opponents.items() if k != "self_play"
                }
                total_prob = sum(
                    prob if isinstance(prob, float) else prob[0]
                    for prob in worker_opponents.values()
                )
                worker_opponents = {
                    key: (
                        value / total_prob
                        if isinstance(value, float)
                        else (value[0] / total_prob, value[1])
                    )
                    for key, value in worker_opponents.items()
                }
                worker_cfg = OpponentsCfg(opponents=worker_opponents)

                env = SelfPlayWarehouseBrawl(
                    reward_manager=reward_manager,
                    opponent_cfg=worker_cfg,
                    save_handler=None,
                    resolution=resolution,
                )

            if fast_mode:
                env.raw_env.max_timesteps = 90

            return env

        return _init

    # ============================================================
    # 4. Vectorized environment setup
    # ============================================================
    if n_envs > 1:
        print(f"Creating {n_envs} parallel environments...")
        env_fns = [make_env(i) for i in range(n_envs)]
        env = SubprocVecEnv(env_fns, start_method="spawn")  # spawn for Windows
    else:
        print("Creating single environment...")
        env = DummyVecEnv([make_env(0)])  # single-env fallback

    # Wrap in VecMonitor for logging (not Gym-only)
    log_dir = f"{save_handler._experiment_path()}/" if save_handler else "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    env = VecMonitor(env, log_dir)

    # ============================================================
    # 5. Initialize agent if needed
    # ============================================================
    if not getattr(agent, "initialized", False):
        probe_env = make_env(0)()
        agent.observation_space = probe_env.observation_space
        agent.obs_helper = probe_env.obs_helper
        agent.action_space = probe_env.action_space
        agent.act_helper = probe_env.act_helper
        agent.env = env
        agent._initialize()
        agent.initialized = True
        probe_env.close()

    # ============================================================
    # 6. CONTINUOUS TRAINING (no epoch resets!)
    # ============================================================
    try:
        print("\n🚀 STARTING CONTINUOUS TRAINING")

        # Calculate total timesteps
        if train_epochs is not None:
            if fast_mode:
                train_timesteps = train_epochs * 90  # 90 timesteps per fast mode game
            else:
                train_timesteps = train_epochs * 2700  # 2700 timesteps per normal game
            print(f"Training for {train_timesteps:,} total timesteps ({train_epochs} games worth)")
        else:
            print(f"Training for {train_timesteps:,} total timesteps")

        print(f"Parallel environments: {n_envs}")
        print(f"Expected wall-clock speedup: ~{n_envs}x (if GPU/compute is bottleneck)")
        print(f"Demo matches will run every {demo_interval_steps:,} timesteps")
        print(f"\n{'='*70}\n")

        # Create demo match callback
        demo_callback = DemoMatchCallback(
            agent=agent,
            save_handler=save_handler,
            demo_interval_steps=demo_interval_steps,
            verbose=1
        )

        # ONE CONTINUOUS LEARN CALL with demo callback - no breaks, no resets, just pure training
        agent.learn(env=env, total_timesteps=train_timesteps, verbose=1, callback=demo_callback)

        print("\n✓ CONTINUOUS TRAINING COMPLETE\n")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
        except:
            pass

        # Stop live visualization
        if visualizer:
            visualizer.stop()
            time.sleep(0.5)
    if train_logging == TrainLogging.PLOT:
        from stable_baselines3.common.results_plotter import load_results, ts2xy
        import matplotlib.pyplot as plt

        try:
            x, y = ts2xy(load_results(log_dir), "timesteps")
            weights = np.repeat(1.0, 50) / 50
            y = np.convolve(y, weights, "valid")
            x = x[len(x) - len(y):]

            fig = plt.figure("Learning Curve")
            plt.plot(x, y)
            plt.xlabel("Number of Timesteps")
            plt.ylabel("Rewards")
            plt.title("Learning Curve (Smoothed)")
            plt.savefig(os.path.join(log_dir, "Learning_Curve.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Plotting failed: {e}")

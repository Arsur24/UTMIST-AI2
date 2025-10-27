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
from stable_baselines3 import PPO

from environment.agent import (
    Agent, SelfPlayWarehouseBrawl, RewardManager,
    SaveHandler, OpponentsCfg, run_match, BasedAgent,
    CameraResolution, TrainLogging
)

import pygame


class LiveGameVisualizer:
    """Real-time game visualization for training"""
    def __init__(self, agent, reward_manager, save_handler, resolution=CameraResolution.LOW):
        self.agent = agent
        self.reward_manager = reward_manager
        self.save_handler = save_handler
        self.resolution = resolution
        self.running = True
        self.latest_timestep = 0

    def start(self):
        """Start visualization in a daemon thread"""
        self.thread = threading.Thread(target=self._run_visualization, daemon=True)
        self.thread.start()

    def _run_visualization(self):
        """Run the actual game visualization during training"""
        try:
            import os
            os.environ['SDL_VIDEODRIVER'] = 'windib'

            from environment.environment import WarehouseBrawl
            from user.train_agent import CustomAgent as VizAgent, MLPExtractor, BasedAgent

            pygame.init()

            # Initialize display
            resolutions = {
                CameraResolution.LOW: (480, 720),
                CameraResolution.MEDIUM: (720, 1280),
                CameraResolution.HIGH: (1080, 1920)
            }
            screen = pygame.display.set_mode(resolutions[self.resolution][::-1])
            pygame.display.set_caption("🎮 Training Visualization")
            clock = pygame.time.Clock()

            # Create visualization match environment (simple, not using save_handler)
            env = WarehouseBrawl(resolution=self.resolution, train_mode=False)

            # Use BasedAgent vs BasedAgent for visualization (no model loading issues)
            viz_agent = BasedAgent()
            opponent = BasedAgent()

            observations, _ = env.reset()
            obs_1 = observations[0]
            obs_2 = observations[1]

            if not viz_agent.initialized: viz_agent.get_env_info(env)
            if not opponent.initialized: opponent.get_env_info(env)

            # Subscribe reward manager signals if available
            if self.reward_manager:
                self.reward_manager.reset()
                self.reward_manager.subscribe_signals(env)

            total_reward = 0.0
            timestep = 0
            frame_count = 0
            last_reward_reason = ""
            last_reward_value = 0.0
            reward_display_timer = 0

            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

                # Get actions
                action_1 = viz_agent.predict(obs_1)
                action_2 = opponent.predict(obs_2)

                # Step environment
                full_action = {0: action_1, 1: action_2}
                observations, rewards, terminated, truncated, info = env.step(full_action)
                obs_1 = observations[0]
                obs_2 = observations[1]

                # Calculate reward using reward_manager
                if self.reward_manager:
                    reward = self.reward_manager.process(env, 1/30.0)

                    # Get the actual reward breakdown from the environment logger
                    log = env.logger[0] if hasattr(env, 'logger') and len(env.logger) > 0 else {}

                    # Determine reward reason from actual values
                    if reward >= 50:
                        last_reward_reason = "on_win_reward: +50"
                    elif reward >= 15:
                        last_reward_reason = "on_drop_reward: +15"
                    elif reward >= 10:
                        last_reward_reason = "on_equip_reward: +10"
                    elif reward >= 8:
                        last_reward_reason = "on_knockout_reward: +8"
                    elif reward >= 5:
                        last_reward_reason = "on_combo_reward: +5"
                    elif reward > 1.0:
                        last_reward_reason = "damage_interaction_reward: +1.0"
                    elif reward > 0.5:
                        last_reward_reason = "danger_zone_reward: +0.5"
                    elif reward > 0:
                        last_reward_reason = f"positive reward: {reward:+.4f}"
                    elif reward < -0.5:
                        last_reward_reason = "danger_zone_reward: -0.5"
                    elif reward < -0.04:
                        last_reward_reason = "penalize_attack_reward: -0.04"
                    elif reward < -0.01:
                        last_reward_reason = "holding_more_than_3_keys: -0.01"
                    elif reward < 0:
                        last_reward_reason = f"penalty: {reward:+.4f}"
                    else:
                        last_reward_reason = "neutral"

                    last_reward_value = reward
                    reward_display_timer = 15  # Display for 15 frames

                    # Print to terminal IMMEDIATELY when reward changes
                    if reward != 0:
                        print(f"🎮 Viz: reward of {reward:+.4f} for {last_reward_reason}")
                else:
                    reward = rewards[0]
                    last_reward_reason = "no_reward_manager"

                total_reward += reward

                # Render game
                try:
                    img = env.render()
                    img_surface = pygame.surfarray.make_surface(img)
                    screen.blit(img_surface, (0, 0))
                except Exception as e:
                    pass

                # Draw reward info on screen
                font = pygame.font.Font(None, 24)
                small_font = pygame.font.Font(None, 20)

                # Current reward
                reward_text = font.render(f"Reward: {reward:+.4f}", True, (0, 255, 0) if reward > 0 else (255, 100, 100))
                total_text = font.render(f"Total: {total_reward:+.2f}", True, (255, 255, 0))
                frame_text = font.render(f"Frame: {frame_count}", True, (100, 200, 255))

                screen.blit(reward_text, (10, 10))
                screen.blit(total_text, (10, 40))
                screen.blit(frame_text, (10, 70))

                # Display reward reason
                if reward_display_timer > 0:
                    reason_text = small_font.render(f"Reason: {last_reward_reason}", True, (200, 200, 100))
                    screen.blit(reason_text, (10, 105))
                    reward_display_timer -= 1


                pygame.display.flip()
                clock.tick(30)

                if terminated or truncated:
                    observations, _ = env.reset()
                    obs_1 = observations[0]
                    obs_2 = observations[1]
                    if self.reward_manager:
                        self.reward_manager.reset()
                    total_reward = 0.0
                    print(f"🎮 Viz: Episode reset at frame {frame_count}")

                frame_count += 1

        except Exception as e:
            print(f"⚠ Visualization error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                pygame.quit()
            except:
                pass

    def stop(self):
        """Stop visualization"""
        self.running = False


def train_parallel(
    agent: Agent,
    reward_manager: RewardManager,
    save_handler: Optional[SaveHandler] = None,
    opponent_cfg: OpponentsCfg = OpponentsCfg(),
    resolution: CameraResolution = CameraResolution.LOW,
    train_timesteps: int = 400_000,
    train_logging: TrainLogging = TrainLogging.PLOT,
    n_envs: int = 10,
    train_epochs: Optional[int] = None,
    fast_mode: bool = False,
    show_live_training: bool = False
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
        n_envs: Number of parallel environments
        train_epochs: Number of epochs (overrides train_timesteps)
        fast_mode: Fast mode for quick testing
        show_live_training: Show live pygame window during training
    """

    # Start live game visualization in background thread
    visualizer = None
    if show_live_training:
        print("\n🎮 Starting live game visualization in background...")
        visualizer = LiveGameVisualizer(agent, reward_manager, save_handler, resolution)
        visualizer.start()
        time.sleep(2)  # Give pygame time to initialize

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
    print(f"Parallel environments: {n_envs}")
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
    # 6. Training loop with epoch-based demo matches
    # ============================================================
    try:
        print("\n🚀 STARTING TRAINING - CustomAgent")

        # Calculate timesteps per epoch
        if train_epochs is not None:
            if fast_mode:
                timesteps_per_epoch = 90  # 90 timesteps per fast mode game
            else:
                timesteps_per_epoch = 2700  # 2700 timesteps per normal game
        else:
            timesteps_per_epoch = train_timesteps

        # Train epoch by epoch
        current_epoch = 0
        total_epochs = train_epochs if train_epochs is not None else 1

        for epoch in range(total_epochs):
            current_epoch = epoch + 1
            print(f"\n{'='*70}")
            print(f"🎮 EPOCH {current_epoch}/{total_epochs} - Training")
            print(f"{'='*70}\n")

            # Train for one epoch
            agent.learn(env=env, total_timesteps=timesteps_per_epoch, verbose=1)

            # Save checkpoint after epoch
            if save_handler:
                save_handler.num_timesteps = agent.get_num_timesteps()
                save_handler.save_agent()
                print(f"✓ Checkpoint saved at epoch {current_epoch}\n")

            # Run demo match after each epoch (ACTUAL GAME RENDERING)
            print(f"\n{'='*70}")
            print(f"🎮 DEMO MATCH - After Epoch {current_epoch}")
            print(f"   (Real game will be displayed)")
            print(f"{'='*70}\n")

            try:
                from environment.agent import run_match
                from user.train_agent import CustomAgent as DemoCustomAgent, MLPExtractor

                # Load the just-trained agent checkpoint
                checkpoint_path = save_handler._checkpoint_path() if save_handler else None
                if checkpoint_path:
                    # Find the latest model file
                    model_files = glob.glob(f"{checkpoint_path}/*.zip")
                    if model_files:
                        latest_model = max(model_files, key=os.path.getctime)
                        print(f"Loading latest checkpoint: {latest_model}")

                        agent_1 = DemoCustomAgent(sb3_class=PPO, extractor=MLPExtractor, file_path=latest_model)
                        agent_2 = DemoCustomAgent(sb3_class=PPO, extractor=MLPExtractor, file_path=latest_model)

                        # Run REAL match with pygame rendering (like demo_match.py)
                        print("Starting match visualization with PYGAME rendering...\n")
                        match_result = run_match(
                            agent_1=agent_1,
                            agent_2=agent_2,
                            max_timesteps=2700,  # Full 90-second match
                            resolution=resolution,
                            video_path=None
                        )

                        print(f"\n✓ Demo Match Complete!")
                        print(f"  Result: {match_result.result}")
                        print(f"  Agent 1 HP: {match_result.player1_hp}")
                        print(f"  Agent 2 HP: {match_result.player2_hp}\n")
            except Exception as demo_error:
                print(f"⚠ Demo match error after epoch {current_epoch}: {demo_error}")
                import traceback
                traceback.print_exc()

        print("\n✓ TRAINING COMPLETE\n")

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

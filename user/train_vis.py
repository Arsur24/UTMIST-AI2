"""
Parallel Training Module for GPU-accelerated training
This module contains train_parallel and related utilities
"""

import os
import time
from functools import partial
from typing import Optional
import threading
import glob

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
            import sys
            # Only force the Windows SDL video driver on Windows platforms.
            # On macOS and Linux the default driver should be used.
            if sys.platform.startswith("win"):
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

            # Enforce real-time stepping (no speed-up): use env.fps if available
            target_fps = getattr(env, 'fps', 30)
            target_dt = 1.0 / target_fps
            last_step_time = time.perf_counter()

            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

                # Throttle stepping to real-time based on target_dt
                now = time.perf_counter()
                elapsed = now - last_step_time
                if elapsed < target_dt:
                    # Still wait until next physics step; but handle display updates and events
                    # Cap the loop rate so we don't spin excessively
                    clock.tick(int(target_fps))
                    continue

                # Perform a single environment step (real-time)
                last_step_time = now

                # Get actions
                action_1 = viz_agent.predict(obs_1)
                action_2 = opponent.predict(obs_2)

                # Step environment
                full_action = {0: action_1, 1: action_2}
                observations, rewards, terminated, truncated, info = env.step(full_action)
                obs_1 = observations[0]
                obs_2 = observations[1]

                # Calculate reward using reward_manager with actual dt
                if self.reward_manager:
                    reward = self.reward_manager.process(env, target_dt)

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
                    reward_display_timer = int(target_fps * 0.5)  # Display for half a second

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
                # Cap display to target fps
                clock.tick(int(target_fps))

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


def run_visualization_only(resolution: CameraResolution = CameraResolution.LOW, run_time: Optional[float] = None):
    """Start the LiveGameVisualizer standalone without running any training or parallel envs.

    This constructs a minimal BasedAgent and RewardManager, launches the visualizer in a thread,
    and blocks until the window is closed or optional run_time (seconds) elapses.
    """
    from environment.agent import BasedAgent, RewardManager

    agent = BasedAgent()
    reward_manager = RewardManager()
    save_handler = None

    visualizer = LiveGameVisualizer(agent, reward_manager, save_handler, resolution=resolution)
    visualizer.start()

    start_ts = time.time()
    try:
        while visualizer.running:
            if run_time is not None and (time.time() - start_ts) >= run_time:
                visualizer.stop()
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        visualizer.stop()
    finally:
        visualizer.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run visualization-only mode for WarehouseBrawl.")
    parser.add_argument("--visualize-only", action="store_true", help="Run only the live visualizer (no training)")
    parser.add_argument("--resolution", choices=[r.name for r in CameraResolution], default=CameraResolution.LOW.name)
    parser.add_argument("--time", type=float, default=None, help="Optional run time in seconds (default: run until window closed)")
    args = parser.parse_args()

    if args.visualize_only:
        res = CameraResolution[args.resolution]
        print(f"Starting visualization-only mode (resolution={res}, time={args.time})")
        run_visualization_only(resolution=res, run_time=args.time)
    else:
        print("This module supports --visualize-only to run only the live visualizer. Exiting.")

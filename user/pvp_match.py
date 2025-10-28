# import skvideo
# import skvideo.io

# Change working directory to project root to fix asset paths
import os
import sys
import traceback
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

from environment.environment import RenderMode
from environment.agent import SB3Agent, CameraResolution, RecurrentPPOAgent, BasedAgent, UserInputAgent, ConstantAgent, run_match, run_real_time_match
from user.my_agent import SubmittedAgent
from user.train_agent import gen_reward_manager

reward_manager = gen_reward_manager()

# ============================================================
# HUMAN VS AI SETUP
# ============================================================
# Player 1: YOU (keyboard controls: WASD, Space, H, J, K, L, G)
my_agent = UserInputAgent()

# Player 2: YOUR TRAINED AI
opponent = SubmittedAgent()  # Loads your trained model from TRAINED_MODEL_PATH

# Use a safe default match time (90 seconds at 30 FPS)
DEFAULT_MATCH_SECONDS = 90
max_timesteps = 30 * DEFAULT_MATCH_SECONDS

if __name__ == '__main__':
    print(f"🎮 HUMAN VS AI MATCH")
    print(f"=" * 60)
    print(f"Player 1 (YOU): {type(my_agent).__name__}")
    print(f"Player 2 (AI):  {type(opponent).__name__}")
    print(f"=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print(f"\n⌨️  CONTROLS:")
    print(f"   WASD - Move/Aim")
    print(f"   SPACE - Jump")
    print(f"   H - Pickup/Throw weapon")
    print(f"   J - Light attack")
    print(f"   K - Heavy attack")
    print(f"   L - Dash/Dodge")
    print(f"   G - Taunt")
    print(f"\n🎯 Starting match...\n")

    try:
        # Run a single real-time match and show pygame window
        result = run_real_time_match(
            agent_1=my_agent,
            agent_2=opponent,
            max_timesteps=max_timesteps,
            resolution=CameraResolution.LOW,
        )
        print(f"\n🏆 MATCH COMPLETE!")
        print(f"Result: {result.player1_result}")
    except Exception as e:
        print("ERROR running match:")
        traceback.print_exc()


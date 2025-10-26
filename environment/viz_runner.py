import argparse
import os
import sys
import time

# Ensure project root is on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from environment.agent import SB3Agent, BasedAgent, UserInputAgent, CameraResolution, run_real_time_match

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to saved model zip')
parser.add_argument('--resolution', default='LOW', help='CameraResolution name')
parser.add_argument('--max_timesteps', type=int, default=30*90)
args = parser.parse_args()

model_path = args.model
res_name = args.resolution

try:
    res = CameraResolution[res_name]
except Exception:
    res = CameraResolution.LOW

# Load model as an agent instance (CustomAgent or SB3Agent compatible interface)
try:
    agent = SB3Agent(file_path=model_path)
except Exception as e:
    print(f"Failed to instantiate SB3Agent for viz: {e}")
    sys.exit(1)

# Opponent
opponent = BasedAgent()

# We run a real-time match (renders with pygame)
# Use UserInputAgent wrapper for agent_1 so run_real_time_match expects a UserInputAgent; however
# run_real_time_match only uses predict; create a lightweight wrapper if necessary.
class VizUserAgent(UserInputAgent):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def predict(self, obs):
        return self.inner.predict(obs)

    def get_env_info(self, env):
        # Delegate to inner agent
        if not getattr(self.inner, 'initialized', False):
            try:
                self.inner.get_env_info(env)
            except Exception:
                pass
        self.initialized = True

viz_agent = VizUserAgent(agent)

# Wait briefly to ensure the main training process can continue
time.sleep(0.5)

try:
    run_real_time_match(viz_agent, opponent, max_timesteps=args.max_timesteps, resolution=res)
except Exception as e:
    print(f"Visualization runner error: {e}")
    raise

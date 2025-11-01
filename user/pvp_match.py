from environment.environment import RenderMode, CameraResolution
from environment.agent import run_real_time_match
from user.train_agent import UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent
from user.my_agent import SubmittedAgent
import pygame
import os
import traceback

# Try to import project_root; if unavailable, derive it relative to this file
try:
    from test_import import project_root
except Exception:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize pygame and ensure working directory is project root
pygame.init()
try:
    # Only change dir if needed
    cwd = os.getcwd()
    if os.path.abspath(cwd) != os.path.abspath(project_root):
        print(f"Changing working directory to project root: {project_root}")
        os.chdir(project_root)
except Exception as e:
    print(f"Warning: couldn't change working directory: {e}")

# Path to model (adjust as needed)
path = "checkpoints/FusedFeatureExtractor6N(DeeperStill)/FusedFeatureExtractor6N(DeeperStill)_1000000_steps.zip"

# Helper to build agent from a checkpoint path (with fallback)
def build_submitted_or_fallback(path):
    if path and os.path.exists(path):
        try:
            return SubmittedAgent(file_path=path)
        except Exception as e:
            print(f"Failed to load SubmittedAgent from {path}: {e}")
            traceback.print_exc()
    else:
        if path:
            print(f"Model not found at: {path} -> falling back to ConstantAgent")
    return ConstantAgent()

# Create agents
my_agent = build_submitted_or_fallback(path)
opponent = build_submitted_or_fallback(path)

# Match time in seconds (sane default)
match_time_seconds = 90  # change this to desired seconds
# Convert to frames (assuming 30 FPS)
FPS = 30
max_frames = int(match_time_seconds * FPS)

# Cap extremely large values to avoid accidental huge runs
MAX_FRAME_CAP = 30 * 60 * 60  # one hour of frames at 30fps
if max_frames > MAX_FRAME_CAP:
    print(f"Capping max_frames ({max_frames}) to {MAX_FRAME_CAP}")
    max_frames = MAX_FRAME_CAP

print(f"Running demo match for {match_time_seconds}s -> {max_frames} frames")

# Run a single real-time match with error handling
try:
    run_real_time_match(
        agent_1=my_agent,
        agent_2=opponent,
        max_timesteps=max_frames,
        resolution=CameraResolution.LOW,
    )
except Exception as e:
    print("Error while running real-time match:")
    traceback.print_exc()

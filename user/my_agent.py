# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import sys
import gdown
import torch
from typing import Optional

# FIX: Change to project root BEFORE importing environment modules
# This prevents "FileNotFoundError: No file 'environment/assets/map/background.png'"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")

# Now safe to import environment modules
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM

# CUDA Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AUTO-DETECT LATEST TRAINED MODEL
def get_latest_checkpoint(checkpoint_dir="checkpoints/experiment_10"):
    """Find the latest trained model checkpoint automatically."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
    if not checkpoints:
        return None

    # Sort by timesteps (extract number from filename)
    def get_steps(filename):
        try:
            return int(filename.split('_')[2].split('.')[0])
        except:
            return 0

    checkpoints.sort(key=get_steps, reverse=True)
    latest = os.path.join(checkpoint_dir, checkpoints[0])
    print(f"🤖 Auto-detected latest model: {latest}")
    return latest

# PATH TO YOUR TRAINED MODEL (auto-detected, or specify manually)
TRAINED_MODEL_PATH = get_latest_checkpoint()  # Auto-finds latest checkpoint!

class SubmittedAgent(Agent):
    '''
    Your trained agent for submission and human vs AI matches!

    Usage:
        # For human vs AI with your trained model:
        agent = SubmittedAgent()  # Loads TRAINED_MODEL_PATH

        # For tournament submission (downloads from Google Drive):
        agent = SubmittedAgent(file_path=None)  # Uses _gdown()
    '''
    def __init__(
        self,
        file_path: Optional[str] = TRAINED_MODEL_PATH,  # Default to trained model
    ):
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            # No path provided - create untrained model (for testing/submission)
            print("⚠ No model path provided - creating UNTRAINED agent")
            self.model = PPO("MlpPolicy", self.env, verbose=0, device=DEVICE)
            del self.env
        else:
            # Load trained model from file_path
            if not os.path.exists(self.file_path):
                print(f"\n❌ ERROR: Model file not found!")
                print(f"   Looking for: {self.file_path}")
                print(f"\n📁 Available checkpoints:")

                # Show all available checkpoints
                checkpoint_dir = os.path.dirname(self.file_path)
                if os.path.exists(checkpoint_dir):
                    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
                    if checkpoints:
                        for f in sorted(checkpoints):
                            print(f"   ✓ {os.path.join(checkpoint_dir, f)}")
                    else:
                        print(f"   (No .zip files found in {checkpoint_dir})")
                else:
                    print(f"   (Directory doesn't exist: {checkpoint_dir})")

                # Try to find any checkpoint as fallback
                print(f"\n🔍 Searching for ANY checkpoint...")
                for root, dirs, files in os.walk("checkpoints"):
                    for f in files:
                        if f.endswith('.zip'):
                            fallback_path = os.path.join(root, f)
                            print(f"\n💡 Found: {fallback_path}")
                            print(f"   Update TRAINED_MODEL_PATH in my_agent.py to use this model")

                raise FileNotFoundError(f"Model not found: {self.file_path}")

            print(f"✅ Loading trained model from: {self.file_path}")
            self.model = PPO.load(self.file_path, device=DEVICE)
            print(f"✅ Model loaded successfully!")

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            # Place a link to your PUBLIC model data here. This is where we will download it from on the tournament server.
            url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    # If modifying the number of models (or training in general), modify this
    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
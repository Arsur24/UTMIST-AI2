'''
TRAINING: AGENT

This file contains all the types of Agent classes, the Reward Function API, and the built-in train function from our multi-agent RL API for self-play training.
- All of these Agent classes are each described below. 

Running this file will initiate the training function, and will:
a) Start training from scratch
b) Continue training from a specific timestep given an input `file_path`
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

# Add project root to sys.path for imports (but don't change working directory yet)
import os
import sys

# NOTE: Do NOT change working directory at import time. We'll change it only when the
# script is executed as __main__ so subprocess imports (workers/visualizers) don't
# repeatedly change/print the working directory.

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

import torch
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER 
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Do NOT call pygame.init() at import time; initialize it inside main when needed.
# pygame will be initialized when running the script directly (see below).

from environment.agent import *
from typing import Optional, Type, List, Tuple

# Import parallel training from user module
from user.parallel_training import train_parallel

# Import reward configuration (robust import that works whether running as a script or package)
try:
    from user import rewards_config as rc
except Exception:
    import rewards_config as rc

# -------------------------------------------------------------------
# -------------------------- CUDA CONFIGURATION ---------------------
# -------------------------------------------------------------------

# Check if CUDA is available and set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NOTE: Do not print device information at import time (subprocesses will import this module).
# If you need to see device info, it will be printed once when running the script directly (__main__).

# -------------------------------------------------------------------
# ----------------------- VISUALIZATION CALLBACK --------------------
# -------------------------------------------------------------------

import time
import subprocess
import sys
from stable_baselines3.common.callbacks import BaseCallback

class VisualizationCallback(BaseCallback):
    """Callback that runs visualization games periodically during training"""
    def __init__(self, train_agent, save_handler, viz_interval_secs, viz_resolution):
        super().__init__()
        self.train_agent = train_agent
        self.save_handler = save_handler
        self.viz_interval_secs = viz_interval_secs
        self.viz_resolution = viz_resolution
        self.last_viz_time = time.time()
        self.viz_count = 0
        self.viz_process = None

    def _on_step(self) -> bool:
        """Called after each training step"""
        current_time = time.time()
        elapsed = current_time - self.last_viz_time

        # Run visualization every N seconds
        if elapsed >= self.viz_interval_secs:
            self.viz_count += 1
            self.last_viz_time = current_time

            print(f"\n{'='*70}")
            print(f"🎮 LAUNCHING VISUALIZATION #{self.viz_count} - Timestep: {self.num_timesteps:,}")
            print(f"{'='*70}")

            try:
                # Save current model to a temp snapshot
                snapshot_path = f"temp_viz_snapshot_{self.viz_count}.zip"
                self.train_agent.model.save(snapshot_path)

                # Kill previous visualization if still running
                if self.viz_process is not None and self.viz_process.poll() is None:
                    self.viz_process.terminate()

                # Launch visualization in a new process
                # Create a simple Python script to run the visualization
                viz_script = f"""
import os
import sys

# Change to project root
os.chdir(r'C:\\Users\\orang\\PycharmProjects\\UTMIST-AI2')
sys.path.insert(0, r'C:\\Users\\orang\\PycharmProjects\\UTMIST-AI2')

# Initialize pygame FIRST before any other imports
import pygame
pygame.init()

from environment.agent import SB3Agent, BasedAgent, run_real_time_match, CameraResolution
from functools import partial
import torch

print("Starting visualization process...")
print(f"Loading model from: {snapshot_path}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the snapshot
try:
    agent = SB3Agent(file_path=r'{snapshot_path}')
    agent.initialized = True
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {{e}}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
    sys.exit(1)

# Run visualization match
try:
    print("Starting match visualization...")
    match_stats = run_real_time_match(
        agent_1=agent,
        agent_2=partial(BasedAgent)(),
        max_timesteps=30*30,
        resolution=CameraResolution.LOW,
        train_mode=False
    )
    print(f"Match result: {{match_stats.player1_result}}")
    print("Match completed!")
except Exception as e:
    print(f"Visualization error: {{e}}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
finally:
    try:
        pygame.quit()
    except:
        pass
"""

                # Write script to temp file
                script_path = f"temp_viz_script_{self.viz_count}.py"
                with open(script_path, 'w') as f:
                    f.write(viz_script)

                # Launch in new process
                print(f"[DEBUG] Attempting to launch visualization subprocess...")
                print(f"[DEBUG] Script path: {script_path}")
                print(f"[DEBUG] Python executable: {sys.executable}")

                self.viz_process = subprocess.Popen(
                    [sys.executable, script_path],
                    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
                )

                print(f"✓ Visualization window launched in new process (PID: {self.viz_process.pid})")
                print(f"{'='*70}\n")

            except Exception as e:
                print(f"⚠ Visualization launch error: {e}")
                import traceback
                traceback.print_exc()

        return True

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    Note:
    - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
    You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            # Choose n_steps and batch_size so batch_size divides (n_steps * n_envs)
            N_STEPS = 30 * 90 * 3
            n_envs = getattr(self.env, 'num_envs', 1)
            computed_batch = N_STEPS * n_envs
            self.model = self.sb3_class(
                "MlpPolicy",
                self.env,
                verbose=0,
                n_steps=N_STEPS,
                batch_size=computed_batch,
                ent_coef=0.01,
                device=DEVICE  # Use CUDA if available
            )
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path, device=DEVICE)

    def _gdown(self) -> str:
        # Call gdown to your link
        return None

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

class RecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 512,
                'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,

            }
            self.model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                verbose=0,
                n_steps=30*90*20,
                batch_size=16,
                ent_coef=0.05,
                policy_kwargs=policy_kwargs,
                device=DEVICE  # Use CUDA if available
            )
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path, device=DEVICE)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class BasedAgent(Agent):
    '''
    BasedAgent:
    - Defines a hard-coded Agent that predicts actions based on if-statements. Interesting behaviour can be achieved here.
    - The if-statement algorithm can be developed within the `predict` method below.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - Defines an Agent that performs actions entirely via real-time player input
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()
       
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)

        return action

class ClockworkAgent(Agent):
    '''
    ClockworkAgent:
    - Defines an Agent that performs sequential steps of [duration, action]
    '''
    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (15, ['space']),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1  # Increment step counter
        return action
    
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, action_dim, dtype=torch.float32)

        # Move model to CUDA if available
        self.to(DEVICE)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        # Ensure input is on the correct device
        if not obs.is_cuda and DEVICE.type == 'cuda':
            obs = obs.to(DEVICE)

        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLPExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP Base Features Extractor
    Note: This should output features (features_dim), NOT actions.
    The policy and value heads will use these features to predict actions/values.
    '''
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        # Create MLP that outputs features_dim (not action_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0], 
            action_dim=features_dim,  # Output features, not actions
            hidden_dim=hidden_dim,
        )
        # Model will be moved to CUDA in MLPPolicy's __init__

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Ensure observations are on the correct device
        if not obs.is_cuda and DEVICE.type == 'cuda':
            obs = obs.to(DEVICE)
        return self.model(obs)
    
    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim)
        )
    
class CustomAgent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: str = None, extractor: Optional[Type[BaseFeaturesExtractor]] = None):
         self.sb3_class = sb3_class
         self.extractor = extractor
         super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            # Compute batch size compatible with n_steps * n_envs
            N_STEPS = 30 * 90 * 3
            n_envs = getattr(self.env, 'num_envs', 1)
            computed_batch = N_STEPS * n_envs
            self.model = self.sb3_class(
                "MlpPolicy",
                self.env,
                policy_kwargs=self.extractor.get_policy_kwargs(),
                verbose=0,
                n_steps=N_STEPS,
                batch_size=computed_batch,
                ent_coef=0.01,
                device=DEVICE  # Use CUDA if available
            )
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path, device=DEVICE)

    def _gdown(self) -> str:
        # Call gdown to your link
        return None

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0, callback=None):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            callback=callback,
        )

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

'''
Example Reward Functions:
- Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).
'''

def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Extract the used quantities (to enable type-hinting)
    obj: GameObject = env.objects[obj_name]

    # Compute the L2 squared penalty
    return (obj.body.position.y - target_height)**2

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def damage_interaction_reward(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward / 140


# In[ ]:


def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    # NOTE: Removed env.dt multiplication to match config weights
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0

    return reward

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState]=BackDashState,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the desired state
    # NOTE: Removed env.dt multiplication to match config weights
    reward = 1 if isinstance(player.state, desired_state) else 0.0

    return reward

def head_to_middle_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def head_to_opponent(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def holding_more_than_3_keys(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    # NOTE: Removed env.dt multiplication to match config weights
    a = player.cur_action
    if (a > 0.5).sum() > 3:
        return 1  # Will be multiplied by weight from config
    return 0

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    else:
        return -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0
    
def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        weapon = env.objects["player"].weapon
        if weapon == "Hammer":
            # Only print from first env to avoid 10x duplication
            if hasattr(env, 'games_done') and env.games_done <= 1:
                print(f"  🎯 on_equip_reward: Equipped Hammer! +2.0")
            return 2.0
        elif weapon == "Spear":
            # Only print from first env
            if hasattr(env, 'games_done') and env.games_done <= 1:
                print(f"  🎯 on_equip_reward: Equipped Spear! +1.0")
            return 1.0
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -1.0
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager():
    # Read weights and params from rewards_config.CFG
    weights = rc.CFG.get('weights', {})
    danger_cfg = rc.CFG.get('danger_zone', {})
    signals = rc.CFG.get('signals', {})

    reward_functions = {
        #'target_height_reward': RewTerm(func=base_height_l2, weight=0.0, params={'target_height': -4, 'obj_name': 'player'}),
        'danger_zone_reward': RewTerm(
            func=danger_zone_reward,
            weight=weights.get('danger_zone_reward', 0.5),
            params={'zone_penalty': danger_cfg.get('zone_penalty', 1), 'zone_height': danger_cfg.get('zone_height', 4.2)}
        ),
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=weights.get('damage_interaction_reward', 1.0)),
        #'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.01),
        #'head_to_opponent': RewTerm(func=head_to_opponent, weight=0.05),
        'penalize_attack_reward': RewTerm(func=in_state_reward, weight=weights.get('penalize_attack_reward', -0.04), params={'desired_state': AttackState}),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=weights.get('holding_more_than_3_keys', -0.01)),
        #'taunt_reward': RewTerm(func=in_state_reward, weight=0.2, params={'desired_state': TauntState}),
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=signals.get('on_win_reward', 50))),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=signals.get('on_knockout_reward', 8))),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=signals.get('on_combo_reward', 5))),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=signals.get('on_equip_reward', 10))),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=signals.get('on_drop_reward', 15)))
    }
    return RewardManager(reward_functions, signal_subscriptions)

# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------
''' 
The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
'''
if __name__ == '__main__':
    # Change working directory to project root to fix asset paths (run-only)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")

    # Print device information once when running as main
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")

    # Add debug print statements to ensure pygame window initialization is logged
    print("Initializing pygame...")
    try:
        # Set SDL to use windib display driver on Windows for better compatibility
        import os
        os.environ['SDL_VIDEODRIVER'] = 'windib'

        pygame.init()
        print("✓ Pygame initialized successfully.")
        print(f"  Pygame version: {pygame.version.ver}")
        try:
            print(f"  Display driver: {pygame.display.get_driver()}")
        except Exception:
            print(f"  Display driver: (could not detect)")
    except Exception as e:
        print(f"✗ Pygame initialization error: {e}")
        print("  Continuing without pygame display support...")
        import traceback
        traceback.print_exc()

    # ===== CUDA CONFIGURATION =====
    # Enable parallel training if CUDA is available
    USE_PARALLEL_TRAINING = torch.cuda.is_available()
    # USE_PARALLEL_TRAINING = False
    N_PARALLEL_ENVS = 10  # Number of parallel environments (use 10 to avoid system instability)

    # Adjust batch size and n_steps for parallel training
    if USE_PARALLEL_TRAINING:
        print(f"\n{'='*60}")
        print(f"CUDA ENABLED - Using GPU-accelerated parallel training")
        print(f"Parallel environments: {N_PARALLEL_ENVS}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"CUDA NOT AVAILABLE - Using CPU training")
        print(f"{'='*60}\n")

    # Create agent
    my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor)

    # Start here if you want to train from scratch. e.g:
    #my_agent = RecurrentPPOAgent()

    # Start here if you want to train from a specific timestep. e.g:
    #my_agent = RecurrentPPOAgent(file_path='checkpoints/experiment_3/rl_model_120006_steps.zip')

    # Reward manager
    reward_manager = gen_reward_manager()
    # Self-play settings
    selfplay_handler = SelfPlayRandom(
        partial(type(my_agent)), # Agent class and its keyword arguments
                                 # type(my_agent) = Agent class
    )

    # Set save settings here:
    save_handler = SaveHandler(
        agent=my_agent, # Agent to save
        save_freq=100_000, # Save frequency
        max_saved=40, # Maximum number of saved models
        save_path='checkpoints', # Save path
        run_name='experiment_10',  # Changed to avoid prompt
        mode=SaveHandlerMode.RESUME # Save mode, FORCE or RESUME
    )

    # Set opponent settings here:
    opponent_specification = {
                    'self_play': (8, selfplay_handler),
                    'constant_agent': (0.5, partial(ConstantAgent)),
                    'based_agent': (1.5, partial(BasedAgent)),
                }
    opponent_cfg = OpponentsCfg(opponents=opponent_specification)

    # Training configuration
    # Quick smoke-test configuration (change back for full runs)
    TRAIN_EPOCHS = 4  # Train for 4 complete games/matches for quick testing
    FAST_MODE = True  # Fast mode: shortened games (90 timesteps) and faster execution
    VISUALIZATION_INTERVAL = 30  # Run visualization game every 30 seconds
    SKIP_INITIAL_VIZ = True  # Skip initial visualization to start training immediately

    # Match durations (frames). Use 30 frames per second as a convention.
    FRAMES_PER_SECOND = 30
    SECONDS_PER_MATCH = 30  # 30 seconds per match
    MATCH_MAX_TIMESTEPS = FRAMES_PER_SECOND * SECONDS_PER_MATCH

    # ===== SHOW INITIAL UNTRAINED AGENT (Optional) =====
    if not SKIP_INITIAL_VIZ:
         print(f"\n{'='*70}")
         print(f"🎮 INITIAL VISUALIZATION - Untrained Agent")
         print(f"{'='*70}")
         try:
             # Use a lightweight preview agent for visualization so we don't construct
             # the SB3 model (which would bind to a single env). RandomAgent represents
             # an "untrained" policy for quick visual checks. We run the real-time
             # pygame match so a window appears.
             preview_agent = RandomAgent()
             # Dynamically import the realtime visualizer to avoid circular import issues
             try:
                 import importlib
                 env_agent = importlib.import_module('environment.agent')
                 rtm = getattr(env_agent, 'run_real_time_match', None)
             except Exception:
                 rtm = None

             if rtm is None:
                 raise ImportError('run_real_time_match not available')

             initial_match = rtm(
                 agent_1=preview_agent,
                 agent_2=BasedAgent(),
                 max_timesteps=MATCH_MAX_TIMESTEPS,
                 resolution=CameraResolution.LOW
             )
             print(f"✓ Initial Real-time Match Result: {initial_match.player1_result}")
             print(f"{'='*70}\n")
         except Exception as e:
             print(f"⚠ Initial visualization error: {e}")
             import traceback
             traceback.print_exc()
    else:
        print(f"\n{'='*70}")
        print(f"⏩ STARTING TRAINING")
        print(f"   Visualization windows will pop up every {VISUALIZATION_INTERVAL}s in separate windows")
        print(f"   Each window shows a match with the current training progress")
        print(f"{'='*70}\n")

    # Use parallel training if CUDA is available, otherwise use standard training
    if USE_PARALLEL_TRAINING:
        train_parallel(
            my_agent,
            reward_manager,
            save_handler,
            opponent_cfg,
            CameraResolution.LOW,
            train_logging=TrainLogging.PLOT,
            n_envs=N_PARALLEL_ENVS,  # Parallel environments for GPU acceleration
            train_epochs=TRAIN_EPOCHS,  # Train for 4 epochs
            fast_mode=FAST_MODE,  # Fast mode for quick testing
            show_live_training=True  # Show real-time game visualization with rewards
        )
    else:
        train(
            my_agent,
            reward_manager,
            save_handler,
            opponent_cfg,
            CameraResolution.LOW,
            train_logging=TrainLogging.PLOT
        )

    print(f"\n{'='*70}")
    print(f"✅ ALL TRAINING AND DEMO MATCHES COMPLETE!")
    print(f"{'='*70}\n")

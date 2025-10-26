from environment.environment import ActHelper, AirTurnaroundState, Animation, AnimationSprite2D, AttackState, BackDashState, Camera, CameraResolution, Capsule, CapsuleCollider, Cast, CastFrameChangeHolder, CasterPositionChange, CasterVelocityDampXY, CasterVelocitySet, CasterVelocitySetXY, CompactMoveState, DashState, DealtPositionTarget, DodgeState, Facing, GameObject, Ground, GroundState, HurtboxPositionChange, InAirState, KOState, KeyIconPanel, KeyStatus, MalachiteEnv, MatchStats, MoveManager, MoveType, ObsHelper, Particle, Player, PlayerInputHandler, PlayerObjectState, PlayerStats, Power, RenderMode, Result, Signal, SprintingState, Stage, StandingState, StunState, Target, TauntState, TurnaroundState, UIHandler, WalkingState, WarehouseBrawl, hex_to_rgb

import warnings
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar, Type, Optional, List, Dict, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, MISSING
from collections import defaultdict
from functools import partial
from typing import Tuple, Any

from PIL import Image, ImageSequence
import matplotlib.pyplot as plt

import gdown, os, math, random, shutil, json

import numpy as np
import torch
from torch import nn

import gymnasium
from gymnasium import spaces

import pygame
import pygame.gfxdraw
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d

import cv2
import skimage.transform as st
import skvideo
import skvideo.io
from IPython.display import Video

from stable_baselines3.common.monitor import Monitor


# ## Agents

# ### Agent Abstract Base Class

# In[ ]:


SelfAgent = TypeVar("SelfAgent", bound="Agent")

class Agent(ABC):

    def __init__(
            self,
            file_path: Optional[str] = None
        ):

        # If no supplied file_path, load from gdown (optional file_path returned)
        if file_path is None:
            file_path = self._gdown()

        self.file_path: Optional[str] = file_path
        self.initialized = False

    def get_env_info(self, env):
        if isinstance(env, Monitor):
            self_env = env.env
        else:
            self_env = env
        self.observation_space = self_env.observation_space
        self.obs_helper = self_env.obs_helper
        self.action_space = self_env.action_space
        self.act_helper = self_env.act_helper
        self.env = env
        self._initialize()
        self.initialized = True

    def get_num_timesteps(self) -> int:
        if hasattr(self, 'model'):
            return self.model.num_timesteps
        else:
            return 0

    def update_num_timesteps(self, num_timesteps: int) -> None:
        if hasattr(self, 'model'):
            self.model.num_timesteps = num_timesteps

    @abstractmethod
    def predict(self, obs) -> spaces.Space:
        pass

    def save(self, file_path: str) -> None:
        return

    def reset(self) -> None:
        return

    def _initialize(self) -> None:
        """

        """
        return

    def _gdown(self) -> Optional[str]:
        """
        Loads the necessary file from Google Drive, returning a file path.
        Or, returns None, if the agent does not require loaded files.

        :return:
        """
        return


# ### Agent Classes

# In[ ]:


class ConstantAgent(Agent):
    '''
    ConstantAgent:
    - The ConstantAgent simply is in an IdleState (action_space all equal to zero.)
    As such it will not do anything, DON'T use this agent for your training.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = np.zeros_like(self.action_space.sample())
        return action

class RandomAgent(Agent):
    '''
    RandomAgent:
    - The RandomAgent (as it name says) simply samples random actions.
    NOT used for training
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.action_space.sample()
        return action


# ## StableBaselines3 Integration

# ### Reward Configuration

# In[ ]:

@dataclass
class RewTerm():
    """Configuration for a reward term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the reward signals as torch float tensors of
    shape (num_envs,).
    """

    weight: float = MISSING
    """The weight of the reward term.

    This is multiplied with the reward term's value to compute the final
    reward.

    Note:
        If the weight is zero, the reward term is ignored.
    """

    params: dict[str, Any] = field(default_factory=dict)
    """The parameters to be passed to the function as keyword arguments. Defaults to an empty dict.

    .. note::
        If the value is a :class:`SceneEntityCfg` object, the manager will query the scene entity
        from the :class:`InteractiveScene` and process the entity's joints and bodies as specified
        in the :class:`SceneEntityCfg` object.
    """



# In[ ]:


class RewardManager():
    """Reward terms for the MDP."""

    # (1) Constant running reward
    def __init__(self,
                 reward_functions: Optional[Dict[str, RewTerm]]=None,
                 signal_subscriptions: Optional[Dict[str, Tuple[str, RewTerm]]]=None) -> None:
        self.reward_functions = reward_functions
        self.signal_subscriptions = signal_subscriptions
        self.total_reward = 0.0
        self.collected_signal_rewards = 0.0

    def subscribe_signals(self, env) -> None:
        if self.signal_subscriptions is None:
            return
        for _, (name, term_cfg) in self.signal_subscriptions.items():
            getattr(env, name).connect(partial(self._signal_func, term_cfg))

    def _signal_func(self, term_cfg: RewTerm, *args, **kwargs):
        term_partial = partial(term_cfg.func, **term_cfg.params)
        self.collected_signal_rewards += term_partial(*args, **kwargs) * term_cfg.weight

    def process(self, env, dt) -> float:
        # reset computation
        reward_buffer = 0.0
        # iterate over all the reward terms
        if self.reward_functions is not None:
            for name, term_cfg in self.reward_functions.items():
                # skip if weight is zero (kind of a micro-optimization)
                if term_cfg.weight == 0.0:
                    continue
                # compute term's value
                value = term_cfg.func(env, **term_cfg.params) * term_cfg.weight
                # update total reward
                reward_buffer += value

        reward = reward_buffer + self.collected_signal_rewards
        self.collected_signal_rewards = 0.0

        self.total_reward += reward

        log = env.logger[0]
        log['reward'] = f'{reward_buffer:.3f}'
        log['total_reward'] = f'{self.total_reward:.3f}'
        env.logger[0] = log
        return reward

    def reset(self):
        self.total_reward = 0
        self.collected_signal_rewards


# ### Save, Self-play, and Opponents

# In[ ]:


class SaveHandlerMode(Enum):
    FORCE = 0
    RESUME = 1

class SaveHandler():
    """Handles saving.

    Args:
        agent (Agent): Agent to save.
        save_freq (int): Number of steps between saving.
        max_saved (int): Maximum number of saved models.
        save_dir (str): Directory to save models.
        name_prefix (str): Prefix for saved models.
    """

    # System for saving to internet

    def __init__(
            self,
            agent: Agent,
            save_freq: int=10_000,
            max_saved: int=20,
            run_name: str='experiment_1',
            save_path: str='checkpoints',
            name_prefix: str = "rl_model",
            mode: SaveHandlerMode=SaveHandlerMode.FORCE
        ):
        self.agent = agent
        self.save_freq = save_freq
        self.run_name = run_name
        self.max_saved = max_saved
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.mode = mode

        self.steps_until_save = save_freq
        # Get model paths from exp_path, if it exists
        exp_path = self._experiment_path()
        self.history: List[str] = []
        if self.mode == SaveHandlerMode.FORCE:
            # Clear old dir
            if os.path.exists(exp_path) and len(os.listdir(exp_path)) != 0:
                while True:
                    answer = input(f"Would you like to clear the folder {exp_path} (SaveHandlerMode.FORCE): yes (y) or no (n): ").strip().lower()
                    if answer in ('y', 'n'):
                        break
                    else:
                        print("Invalid input, please enter 'y' or 'n'.")

                if answer == 'n':
                    # Switch to RESUME mode and continue from existing checkpoint
                    print(f'Switching to RESUME mode - continuing from existing checkpoint in {exp_path}...')
                    self.mode = SaveHandlerMode.RESUME
                    # Get all model paths
                    self.history = [os.path.join(exp_path, f) for f in os.listdir(exp_path) if os.path.isfile(os.path.join(exp_path, f))]
                    # Filter any non .zip files
                    self.history = [f for f in self.history if f.endswith('.zip')]
                    if len(self.history) != 0:
                        self.history.sort(key=lambda x: int(os.path.basename(x).split('_')[-2].split('.')[0]))
                        if max_saved != -1: self.history = self.history[-max_saved:]
                        print(f'Found {len(self.history)} existing checkpoint(s). Latest: {os.path.basename(self.history[-1])}')
                    else:
                        print(f'No checkpoints found in {exp_path}. Starting fresh training.')
                else:
                    print(f'Clearing {exp_path}...')
                    if os.path.exists(exp_path):
                        shutil.rmtree(exp_path)
            else:
                print(f'{exp_path} empty or does not exist. Creating...')

            if not os.path.exists(exp_path):
                os.makedirs(exp_path)
        elif self.mode == SaveHandlerMode.RESUME:
            if os.path.exists(exp_path):
                # Get all model paths
                self.history = [os.path.join(exp_path, f) for f in os.listdir(exp_path) if os.path.isfile(os.path.join(exp_path, f))]
                # Filter any non .csv
                self.history = [f for f in self.history if f.endswith('.zip')]
                if len(self.history) != 0:
                    self.history.sort(key=lambda x: int(os.path.basename(x).split('_')[-2].split('.')[0]))
                    if max_saved != -1: self.history = self.history[-max_saved:]
                    print(f'Best model is {self.history[-1]}')
                else:
                    print(f'No models found in {exp_path}.')
                    raise FileNotFoundError
            else:
                print(f'No file found at {exp_path}')


    def update_info(self) -> None:
        self.num_timesteps = self.agent.get_num_timesteps()

    def _experiment_path(self) -> str:
        """
        Helper to get experiment path for each type of checkpoint.

        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, self.run_name)

    def _checkpoint_path(self, extension: str = '') -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self._experiment_path(), f"{self.name_prefix}_{self.num_timesteps}_steps.{extension}")

    def save_agent(self) -> None:
        print(f"Saving agent to {self._checkpoint_path()}")
        model_path = self._checkpoint_path('zip')
        self.agent.save(model_path)
        self.history.append(model_path)
        if self.max_saved != -1 and len(self.history) > self.max_saved:
            os.remove(self.history.pop(0))

    def process(self) -> bool:
        self.num_timesteps += 1

        if self.steps_until_save <= 0:
            # Save agent
            self.steps_until_save = self.save_freq
            self.save_agent()
            return True
        self.steps_until_save -= 1

        return False

    def get_random_model_path(self) -> str:
        if len(self.history) == 0:
            return None
        return random.choice(self.history)

    def get_latest_model_path(self) -> str:
        if len(self.history) == 0:
            return None
        return self.history[-1]

class SelfPlayHandler(ABC):
    """Handles self-play."""

    def __init__(self, agent_partial: partial):
        self.agent_partial = agent_partial
    
    def get_model_from_path(self, path) -> Agent:
        if path:
            try:
                opponent = self.agent_partial(file_path=path)
            except FileNotFoundError:
                print(f"Warning: Self-play file {path} not found. Defaulting to constant agent.")
                opponent = ConstantAgent()
        else:
            print("Warning: No self-play model saved. Defaulting to constant agent.")
            opponent = ConstantAgent()
        opponent.get_env_info(self.env)
        return opponent

    @abstractmethod
    def get_opponent(self) -> Agent:
        pass

class SelfPlayLatest(SelfPlayHandler):
    def __init__(self, agent_partial: partial):
        super().__init__(agent_partial)
    
    def get_opponent(self) -> Agent:
        assert self.save_handler is not None, "Save handler must be specified for self-play"
        chosen_path = self.save_handler.get_latest_model_path()
        return self.get_model_from_path(chosen_path)

class SelfPlayRandom(SelfPlayHandler):
    def __init__(self, agent_partial: partial):
        super().__init__(agent_partial)
    
    def get_opponent(self) -> Agent:
        assert self.save_handler is not None, "Save handler must be specified for self-play"
        chosen_path = self.save_handler.get_random_model_path()
        return self.get_model_from_path(chosen_path)

@dataclass
class OpponentsCfg():
    """Configuration for opponents.

    Args:
        swap_steps (int): Number of steps between swapping opponents.
        opponents (dict): Dictionary specifying available opponents and their selection probabilities.
    """
    swap_steps: int = 10_000
    opponents: dict[str, Any] = field(default_factory=lambda: {
                'random_agent': (0.8, partial(RandomAgent)),
                'constant_agent': (0.2, partial(ConstantAgent)),
                #'recurrent_agent': (0.1, partial(RecurrentPPOAgent, file_path='skibidi')),
            })

    def validate_probabilities(self) -> None:
        total_prob = sum(prob if isinstance(prob, float) else prob[0] for prob in self.opponents.values())

        if abs(total_prob - 1.0) > 1e-5:
            print(f"Warning: Probabilities do not sum to 1 (current sum = {total_prob}). Normalizing...")
            self.opponents = {
                key: (value / total_prob if isinstance(value, float) else (value[0] / total_prob, value[1]))
                for key, value in self.opponents.items()
            }

    def process(self) -> None:
        pass

    def on_env_reset(self) -> Agent:

        agent_name = random.choices(
            list(self.opponents.keys()),
            weights=[prob if isinstance(prob, float) else prob[0] for prob in self.opponents.values()]
        )[0]

        # If self-play is selected, return the trained model
        print(f'Selected {agent_name}')
        if agent_name == "self_play":
            selfplay_handler: SelfPlayHandler = self.opponents[agent_name][1]
            return selfplay_handler.get_opponent()
        else:
            # Otherwise, return an instance of the selected agent class
            opponent = self.opponents[agent_name][1]()

        opponent.get_env_info(self.env)
        return opponent


# ### Self-Play Warehouse Brawl

# In[ ]:


class SelfPlayWarehouseBrawl(gymnasium.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 reward_manager: Optional[RewardManager]=None,
                 opponent_cfg: OpponentsCfg=OpponentsCfg(),
                 save_handler: Optional[SaveHandler]=None,
                 render_every: int | None = None,
                 resolution: CameraResolution=CameraResolution.LOW):
        """
        Initializes the environment.

        Args:
            reward_manager (Optional[RewardManager]): Reward manager.
            opponent_cfg (OpponentCfg): Configuration for opponents.
            save_handler (SaveHandler): Configuration for self-play.
            render_every (int | None): Number of steps between a demo render (None if no rendering).
        """
        super().__init__()

        self.reward_manager = reward_manager
        self.save_handler = save_handler
        self.opponent_cfg = opponent_cfg
        self.render_every = render_every
        self.resolution = resolution

        self.games_done = 0


        # Give OpponentCfg references, and normalize probabilities
        self.opponent_cfg.env = self
        self.opponent_cfg.validate_probabilities()

        # Check if using self-play
        for key, value in self.opponent_cfg.opponents.items():
            if isinstance(value[1], SelfPlayHandler):
                assert self.save_handler is not None, "Save handler must be specified for self-play"

                # Give SelfPlayHandler references
                selfplay_handler: SelfPlayHandler = value[1]
                selfplay_handler.save_handler = self.save_handler
                selfplay_handler.env = self       

        self.raw_env = WarehouseBrawl(resolution=resolution, train_mode=True)
        self.action_space = self.raw_env.action_space
        self.act_helper = self.raw_env.act_helper
        self.observation_space = self.raw_env.observation_space
        self.obs_helper = self.raw_env.obs_helper

    def on_training_start(self):
        # Update SaveHandler
        if self.save_handler is not None:
            self.save_handler.update_info()

    def on_training_end(self):
        if self.save_handler is not None:
            self.save_handler.agent.update_num_timesteps(self.save_handler.num_timesteps)
            self.save_handler.save_agent()

    def step(self, action):

        full_action = {
            0: action,
            1: self.opponent_agent.predict(self.opponent_obs),
        }

        observations, rewards, terminated, truncated, info = self.raw_env.step(full_action)

        if self.save_handler is not None:
            self.save_handler.process()

        if self.reward_manager is None:
            reward = rewards[0]
        else:
            reward = self.reward_manager.process(self.raw_env, 1 / 30.0)

        return observations[0], reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset MalachiteEnv
        observations, info = self.raw_env.reset()

        # Guard against None reward_manager (subprocess envs may not receive a RM)
        if self.reward_manager is not None:
            try:
                self.reward_manager.reset()
            except Exception as e:
                print(f"Warning: reward_manager.reset() failed: {e}")
 
        # Select agent
        new_agent: Agent = self.opponent_cfg.on_env_reset()
        if new_agent is not None:
            self.opponent_agent: Agent = new_agent
        self.opponent_obs = observations[1]


        self.games_done += 1
        #if self.games_done % self.render_every == 0:
            #self.render_out_video()

        return observations[0], info

    def render(self):
        img = self.raw_env.render()
        return img

    def close(self):
        pass


# ## Run Match

# In[ ]:


from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from tqdm import tqdm

def run_match(agent_1: Agent | partial,
              agent_2: Agent | partial,
              max_timesteps=30*90,
              video_path: Optional[str]=None,
              agent_1_name: Optional[str]=None,
              agent_2_name: Optional[str]=None,
              resolution = CameraResolution.LOW,
              reward_manager: Optional[RewardManager]=None,
              train_mode=False
              ) -> MatchStats:
    # Initialize env

    env = WarehouseBrawl(resolution=resolution, train_mode=train_mode)
    observations, infos = env.reset()
    obs_1 = observations[0]
    obs_2 = observations[1]
    print("RUN MATCH IS RUNNING")
    if reward_manager is not None:
        reward_manager.reset()
        reward_manager.subscribe_signals(env)

    if agent_1_name is None:
        agent_1_name = 'agent_1'
    if agent_2_name is None:
        agent_2_name = 'agent_2'

    env.agent_1_name = agent_1_name
    env.agent_2_name = agent_2_name


    writer = None
    if video_path is None:
        print("video_path=None -> Not rendering")
    else:
        print(f"video_path={video_path} -> Rendering")
        # Initialize video writer
        writer = skvideo.io.FFmpegWriter(video_path, outputdict={
            '-vcodec': 'libx264',  # Use H.264 for Windows Media Player
            '-pix_fmt': 'yuv420p',  # Compatible with both WMP & Colab
            '-preset': 'fast',  # Faster encoding
            '-crf': '20',  # Quality-based encoding (lower = better quality)
            '-r': '30'  # Frame rate
        })

    # If partial
    if callable(agent_1):
        agent_1 = agent_1()
    if callable(agent_2):
        agent_2 = agent_2()

    # Initialize agents
    if not agent_1.initialized: agent_1.get_env_info(env)
    if not agent_2.initialized: agent_2.get_env_info(env)
    # 596, 336
    platform1 = env.objects["platform1"]

    for time in tqdm(range(max_timesteps), total=max_timesteps):
      platform1.physics_process(0.05)
      full_action = {
          0: agent_1.predict(obs_1),
          1: agent_2.predict(obs_2)
      }

      observations, rewards, terminated, truncated, info = env.step(full_action)
      obs_1 = observations[0]
      obs_2 = observations[1]

      if reward_manager is not None:
          reward_manager.process(env, 1 / env.fps)

      if video_path is not None:
          img = env.render()
          writer.writeFrame(img)
          del img

      if terminated or truncated:
          break


    if video_path is not None:
        writer.close()
    env.close()

    # visualize
    # Video(video_path, embed=True, width=800) if video_path is not None else None
    player_1_stats = env.get_stats(0)
    player_2_stats = env.get_stats(1)

    if player_1_stats.lives_left > player_2_stats.lives_left:
        result = Result.WIN
    elif player_1_stats.lives_left < player_2_stats.lives_left:
        result = Result.LOSS
    else:
        result = Result.DRAW
    
    match_stats = MatchStats(
        match_time=env.steps / env.fps,
        player1=player_1_stats,
        player2=player_2_stats,
        player1_result=result
    )

    del env

    return match_stats


class ConstantAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = np.zeros_like(self.action_space.sample())
        return action

class RandomAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.action_space.sample()
        return action


class BasedAgent(Agent):

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

        #if keys[pygame.K_q]:
        #    action = self.act_helper.press_keys(['q'], action)
        #if keys[pygame.K_v]:
        #    action = self.act_helper.press_keys(['v'], action)
        return action


class ClockworkAgent(Agent):

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
                (30, []),
                (7, ['d']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (20, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
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

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

class SB3Agent(Agent):

    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

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

from sb3_contrib import RecurrentPPO

# CUDA Configuration for environment agents
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RecurrentPPOAgent(Agent):

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


# ## Training Function
# A helper function for training.

# In[ ]:


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

class TrainLogging(Enum):
    NONE = 0
    TO_FILE = 1
    PLOT = 2

def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")

    weights = np.repeat(1.0, 50) / 50
    print(weights, y)
    y = np.convolve(y, weights, "valid")
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")

    # save to file
    plt.savefig(log_folder + title + ".png")

def train(agent: Agent,
          reward_manager: RewardManager,
          save_handler: Optional[SaveHandler]=None,
          opponent_cfg: OpponentsCfg=OpponentsCfg(),
          resolution: CameraResolution=CameraResolution.LOW,
          train_timesteps: int=400_000,
          train_logging: TrainLogging=TrainLogging.PLOT,
          train_epochs: Optional[int]=None,
          fast_mode: bool=False
          ):
    """
    Train an agent with either timesteps or epochs.

    Args:
        agent: Agent to train
        reward_manager: Reward manager
        save_handler: Save handler for checkpoints
        opponent_cfg: Opponent configuration
        resolution: Camera resolution
        train_timesteps: Total timesteps to train (ignored if train_epochs is set)
        train_logging: Logging mode
        train_epochs: Number of complete games/epochs to train (overrides train_timesteps if set)
        fast_mode: If True, runs games at 10x speed for quick testing (10 games per second)
    """
    # Create environment
    env = SelfPlayWarehouseBrawl(reward_manager=reward_manager,
                                 opponent_cfg=opponent_cfg,
                                 save_handler=save_handler,
                                 resolution=resolution
                                 )
    reward_manager.subscribe_signals(env.raw_env)

    # Apply fast mode settings
    if fast_mode:
        # Reduce game length to 3 seconds and speed up to 10x
        env.raw_env.max_timesteps = 90  # 3 seconds at 30 FPS = 90 timesteps per game
        print(f"\n{'='*60}")
        print(f"FAST MODE ENABLED - 10 games per second")
        print(f"Each game: 90 timesteps (3 seconds)")
        print(f"{'='*60}\n")

    if train_logging != TrainLogging.NONE:
        # Create log dir
        log_dir = f"{save_handler._experiment_path()}/" if save_handler is not None else "/tmp/gym/"
        os.makedirs(log_dir, exist_ok=True)

        # Logs will be saved in log_dir/monitor.csv
        env = Monitor(env, log_dir)

    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env

    # Calculate timesteps based on epochs if specified
    if train_epochs is not None:
        if fast_mode:
            # In fast mode, each game is 90 timesteps
            train_timesteps = train_epochs * 90
        else:
            # Normal mode: average game is ~30*90 = 2700 timesteps
            train_timesteps = train_epochs * 2700
        print(f"\n{'='*60}")
        print(f"Training for {train_epochs} epochs (approximately {train_timesteps:,} timesteps)")
        print(f"{'='*60}\n")

    try:
        agent.get_env_info(base_env)
        base_env.on_training_start()

        # Create custom callback for visualization
        from stable_baselines3.common.callbacks import BaseCallback

        class VisualizationCallback(BaseCallback):
            """Callback that runs visualization games periodically"""
            def __init__(self, train_agent, viz_opponent, viz_interval_secs, viz_resolution):
                super().__init__()
                self.train_agent = train_agent
                self.viz_opponent = viz_opponent
                self.viz_interval_secs = viz_interval_secs
                self.viz_resolution = viz_resolution
                self.last_viz_time = time.time()
                self.viz_count = 0

            def _on_step(self) -> bool:
                """Called after each training step"""
                current_time = time.time()
                elapsed = current_time - self.last_viz_time

                # Run visualization every N seconds
                if elapsed >= self.viz_interval_secs:
                    self.viz_count += 1
                    print(f"\n{'='*70}")
                    print(f"🎮 VISUALIZATION #{self.viz_count} - Timestep: {self.num_timesteps:,}")
                    print(f"{'='*70}")

                    try:
                        # Run demo match
                        match_stats = run_match(
                            agent_1=self.train_agent,
                            agent_2=self.viz_opponent,
                            max_timesteps=30*90,
                            agent_1_name="Trained Agent",
                            agent_2_name="Opponent",
                            resolution=self.viz_resolution,
                            train_mode=True
                        )

                        print(f"✓ Match Result: {match_stats.player1_result}")
                        print(f"{'='*70}\n")
                    except Exception as e:
                        print(f"⚠ Visualization error: {e}")

                    self.last_viz_time = current_time

                return True

        # Create visualization callback
        viz_callback = VisualizationCallback(
            train_agent=agent,
            viz_opponent=partial(BasedAgent)(),
            viz_interval_secs=90,
            viz_resolution=resolution
        )

        agent.learn(env, total_timesteps=train_timesteps, verbose=1, callback=viz_callback)
        base_env.on_training_end()
    except KeyboardInterrupt:
        pass

    env.close()

    if save_handler is not None:
        save_handler.save_agent()

    if train_logging == TrainLogging.PLOT:
        plot_results(log_dir)


def train_parallel(agent: Agent,
          reward_manager: RewardManager,
          save_handler: Optional[SaveHandler]=None,
          opponent_cfg: OpponentsCfg=OpponentsCfg(),
          resolution: CameraResolution=CameraResolution.LOW,
          train_timesteps: int=400_000,
          train_logging: TrainLogging=TrainLogging.PLOT,
          n_envs: int=4,
          train_epochs: Optional[int]=None,
          fast_mode: bool=False,
          visualization_interval: int=90,
          custom_callback: Optional['BaseCallback']=None
          ):
    """
    Enhanced training function with parallel environment support for better GPU utilization.

    Args:
        agent: The agent to train
        reward_manager: Manager for reward calculations
        save_handler: Handler for saving checkpoints
        opponent_cfg: Configuration for opponents
        resolution: Camera resolution
        train_timesteps: Total timesteps to train (ignored if train_epochs is set)
        train_logging: Logging mode
        n_envs: Number of parallel environments (default: 4 for better GPU utilization)
        train_epochs: Number of complete games/epochs to train (overrides train_timesteps if set)
        fast_mode: If True, runs games at 10x speed for quick testing (10 games per second per env)
        visualization_interval: Seconds between visualization games (default: 90 seconds)
    """
    import torch
    import time
    from threading import Thread

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of parallel environments: {n_envs}")

    # Calculate timesteps based on epochs if specified
    if train_epochs is not None:
        if fast_mode:
            # In fast mode, each game is 90 timesteps
            train_timesteps = train_epochs * 90
            print(f"FAST MODE: Training for {train_epochs} epochs (approximately {train_timesteps:,} timesteps)")
            print(f"Expected completion: ~{train_epochs / (10 * n_envs):.1f} seconds with {n_envs} parallel envs")
        else:
            # Normal mode: average game is ~30*90 = 2700 timesteps
            train_timesteps = train_epochs * 2700
            print(f"Training for {train_epochs} epochs (approximately {train_timesteps:,} timesteps)")

    # Visualization function
    def run_visualization(trained_agent, opponent_agent, viz_resolution):
        """Run a demo game with current trained agent"""
        try:
            print(f"\n{'='*60}")
            print(f"▶ RUNNING VISUALIZATION GAME")
            print(f"{'='*60}")

            from environment.agent import run_match, CameraResolution

            # Run a quick match with current agent
            match_stats = run_match(
                agent_1=trained_agent,
                agent_2=opponent_agent,
                max_timesteps=30*90,  # Full 90-second game
                agent_1_name="Trained Agent",
                agent_2_name="Opponent",
                resolution=viz_resolution,
                train_mode=True
            )

            print(f"\n{'='*60}")
            print(f"✓ VISUALIZATION COMPLETE")
            print(f"Match Result: {match_stats.player1_result}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Visualization error: {e}")

    # Create opponent for visualization
    viz_opponent = partial(BasedAgent)()

    # Track time for visualization
    last_viz_time = time.time()
    visualization_interval_seconds = visualization_interval

    # --- Create a lightweight probe environment (non-vectorized) to initialize agent env info ---
    # We need a concrete env (not a VecEnv) so Agent.get_env_info can read attributes like
    # observation_space, obs_helper and act_helper. Construct a full SelfPlayWarehouseBrawl
    # with the provided reward_manager/opponent_cfg/save_handler for probing, then close it.
    try:
        probe_env = SelfPlayWarehouseBrawl(
            reward_manager=reward_manager,
            opponent_cfg=opponent_cfg,
            save_handler=save_handler,
            resolution=resolution
        )
        # Subscribe reward manager signals to the raw environment so reward terms work during training
        if reward_manager is not None:
            reward_manager.subscribe_signals(probe_env.raw_env)

        # Instead of calling agent.get_env_info(probe_env) which may _initialize the SB3 model
        # with a single-environment env, copy only the helper attributes required for agent logic
        # and postpone actual SB3 model creation until after the vectorized env is created.
        agent.observation_space = probe_env.observation_space
        agent.obs_helper = probe_env.obs_helper
        agent.action_space = probe_env.action_space
        agent.act_helper = probe_env.act_helper
        agent.env = None
        agent.initialized = False
    finally:
        # Close probe env to release pygame/resources before launching subprocesses
        try:
            probe_env.close()
        except Exception:
            pass

    # Create environment factory function
    def make_env():
        def _init():
            # Create a lightweight SelfPlayWarehouseBrawl inside the subprocess to avoid capturing
            # large or unpicklable objects (pygame surfaces, handlers) in the closure.
            env = SelfPlayWarehouseBrawl(
                # Do not pass reward_manager, opponent_cfg, or save_handler here so the factory
                # function remains picklable. These can be set later if needed.
                resolution=resolution
            )
            # Apply fast mode settings
            if fast_mode:
                env.raw_env.max_timesteps = 90  # 3 seconds at 30 FPS = 90 timesteps per game
            return env
        return _init

    # Create vectorized environment for parallel training
    if n_envs > 1 and torch.cuda.is_available():
        print(f"Creating {n_envs} parallel environments for GPU-accelerated training...")
        # Use top-level factory via functools.partial so the callables are picklable
        env_fns = [partial(_create_selfplay_env, resolution=resolution, fast_mode=fast_mode) for _ in range(n_envs)]
        env = SubprocVecEnv(env_fns)
    else:
        print("Creating single environment (CPU mode or n_envs=1)...")
        env = _create_selfplay_env(resolution, fast_mode)

    # Now that we have the (possibly vectorized) env, create the SB3 model using that env
    # if the agent has not already created a model. This ensures the model's n_envs matches.
    try:
        desired_n_envs = getattr(env, 'num_envs', 1)
        if hasattr(agent, 'model') and getattr(agent, 'model', None) is not None:
            current_n_envs = getattr(agent.model, 'n_envs', 1)
            if current_n_envs != desired_n_envs:
                print(f"Agent model has {current_n_envs} env(s) but trainer needs {desired_n_envs}; recreating model...")
                try:
                    # Safely remove old model and create a new one bound to the vectorized env
                    delattr(agent, 'model')
                except Exception:
                    agent.model = None
                agent.env = env
                if hasattr(agent, '_initialize'):
                    agent._initialize()
                agent.initialized = True
        else:
            # assign the env so _initialize can build the model correctly
            agent.env = env
            # Call the agent-specific initialization which should construct/load the SB3 model
            if hasattr(agent, '_initialize'):
                agent._initialize()
            agent.initialized = True
    except Exception as e:
        print(f"Error initializing agent model with vectorized env: {e}")
        raise

    # Subscribe reward manager signals
    if reward_manager is not None:
        if hasattr(env, 'raw_env'):
            reward_manager.subscribe_signals(env.raw_env)
        elif hasattr(env, 'envs') and len(env.envs) > 0:
            # For vectorized environments, subscribe to first env
            reward_manager.subscribe_signals(env.envs[0].raw_env)

    if train_logging != TrainLogging.NONE:
        # Create log dir
        log_dir = f"{save_handler._experiment_path()}/" if save_handler is not None else "/tmp/gym/"
        os.makedirs(log_dir, exist_ok=True)

        # For vectorized environments, wrap with VecMonitor
        if n_envs > 1:
            from stable_baselines3.common.vec_env import VecMonitor
            env = VecMonitor(env, log_dir)
        else:
            env = Monitor(env, log_dir)

    base_env = env.unwrapped if hasattr(env, 'unwrapped') else (env.envs[0] if hasattr(env, 'envs') else env)

    try:
        # Agent already initialized with a concrete probe environment earlier.
        # Do not call get_env_info on a vectorized env (e.g., SubprocVecEnv) because
        # it doesn't expose attributes like `obs_helper`. Proceed directly to learning.
        agent.learn(env, total_timesteps=train_timesteps, verbose=1)
        
        # Start a background visualization thread that periodically saves the
        # current model and launches a separate process to render a real-time
        # match. This keeps training running uninterrupted while providing a
        # concurrent visualization.
        viz_stop_event = None
        viz_thread = None
        import threading
        if custom_callback is None and save_handler is not None and visualization_interval is not None and visualization_interval > 0:
            import tempfile, subprocess, sys
            viz_stop_event = threading.Event()

            def viz_worker():
                while not viz_stop_event.wait(visualization_interval):
                    try:
                        # Save a temporary snapshot of the model
                        tmpdir = tempfile.mkdtemp()
                        model_path = os.path.join(tmpdir, 'viz_model.zip')
                        try:


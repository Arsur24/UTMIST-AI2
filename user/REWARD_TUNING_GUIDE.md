# Reward Tuning Guide for Improved Learning

## Overview
This guide documents the improvements made to the RL training reward system to achieve better learning curves and faster convergence.

## Problem Analysis
The original learning curve showed:
- Rewards in the -15 to -21 range (all negative)
- Slow improvement over 160k timesteps
- Heavy penalties dominating sparse positive rewards
- Low exploration coefficient limiting agent behavior diversity

## Changes Made

### 1. Reward Balance Improvements (`rewards_config.json`)

#### Cliff/Danger Zone Penalty
- **Before**: Weight = 0.0 (disabled) or -50.0 (too harsh per problem statement)
- **After**: Weight = -20.0 (active but reasonable)
- **Zone Height**: Lowered from 4.2 to 3.5 (trigger penalty earlier)
- **Rationale**: -50 per frame was too punishing. -20 discourages cliff jumping while allowing learning.

#### Attack Spam Penalty
- **Before**: -0.04 per frame in attack state
- **After**: -0.02 per frame (halved)
- **Rationale**: Original penalty was too harsh, discouraging necessary combat engagement.

#### Weapon Drop Penalty
- **Before**: -20 per drop event
- **After**: -5 per drop event
- **Rationale**: -20 was extremely punishing. -5 still discourages drops but allows recovery from mistakes.

#### Positive Reward Increases
| Reward Type | Before | After | Reason |
|------------|--------|-------|---------|
| Knockout | 8 | 10 | Encourage aggressive play |
| Combo | 5 | 8 | Reward skillful fighting |
| Weapon Equip | 10 | 15 | Encourage weapon usage |
| Win | 50 | 50 | Unchanged (already good) |

#### New: Survival Reward
- **Weight**: +0.1 per frame alive
- **Purpose**: Provides continuous positive signal to balance penalties
- **Impact**: Over 2700 frames (90s match), this gives up to +270 reward for survival
- **Rationale**: Counters the dominance of negative penalties, gives agent incremental feedback

### 2. Hyperparameter Improvements (`train_agent.py`)

#### Exploration Coefficient (ent_coef)
- **Before**: 0.01 (too conservative)
- **After**: 0.05 (5x increase)
- **Applied To**: SB3Agent, CustomAgent
- **Impact**: Encourages more diverse behavior exploration during training
- **Rationale**: Low exploration caused agent to get stuck in local minima

#### Learning Rate
- **Before**: Default (implicit, likely 3e-4)
- **After**: 3e-4 (explicit)
- **Rationale**: Making it explicit ensures consistent training behavior

#### Training Duration
- **Before**: 100,000 timesteps
- **After**: 500,000 timesteps (5x increase)
- **Rationale**: 100k was too short for convergence. 500k-1M recommended for good results.

### 3. Reward Function Addition

```python
def survival_reward(env: WarehouseBrawl) -> float:
    """
    Provides a small positive reward every frame to encourage staying alive.
    Balances negative penalties and provides incremental learning signal.
    """
    player: Player = env.objects["player"]
    is_alive = not (hasattr(player.state, '__class__') and 
                    player.state.__class__.__name__ in ['KnockedOutState', 'FallenState'])
    return 1.0 if is_alive else 0.0
```

Added to `gen_reward_manager()` with weight from config (0.1).

## Expected Improvements

### Learning Curve
- **Before**: Rewards -15 to -21, slow improvement
- **Expected After**: 
  - Starting rewards: -5 to -10 (less negative)
  - After 100k steps: 0 to +10 (positive territory)
  - After 500k steps: +15 to +30 (solid performance)

### Agent Behavior
1. **More Exploration**: Higher ent_coef leads to trying more strategies
2. **Better Combat**: Reduced attack penalty encourages fighting
3. **Weapon Usage**: Higher equip reward motivates weapon pickup
4. **Cliff Awareness**: -20 penalty still teaches avoidance without being crippling
5. **Survival Focus**: Continuous +0.1/frame encourages staying alive

## Tuning Recommendations

### If Agent is Too Conservative
- Increase `damage_interaction_reward` weight to 1.5
- Increase `on_knockout_reward` to 15
- Decrease `danger_zone_reward` to -10

### If Agent Jumps Off Cliffs Too Much
- Increase `danger_zone_reward` to -30 or -40
- Lower `zone_height` to 3.0 (trigger earlier)

### If Agent Drops Weapons Too Often
- Increase `on_drop_reward` penalty to -10 or -15
- Increase `on_equip_reward` to 20

### If Learning is Too Slow
- Increase `learning_rate` to 5e-4
- Increase `ent_coef` to 0.08
- Train for 1M timesteps instead of 500k

### If Agent is Too Aggressive/Reckless
- Increase `survival_reward` to 0.15
- Decrease `damage_interaction_reward` to 0.7
- Add small penalty for taking damage in damage_interaction_reward

## Monitoring Training

### Good Signs
- Rewards trending from negative to positive over time
- Learning curve smoothly increasing (even if noisy)
- Demo matches showing improved combat skills
- Agent surviving longer in matches

### Warning Signs
- Rewards stuck at same level for >100k steps → increase exploration
- Rewards decreasing → check for bugs or reward conflicts
- Agent repeating same failed strategy → increase ent_coef or learning_rate
- High variance with no trend → may need more training time

## Configuration Files

### rewards_config.json
```json
{
  "weights": {
    "danger_zone_reward": -20.0,
    "damage_interaction_reward": 1.0,
    "penalize_attack_reward": -0.02,
    "holding_more_than_3_keys": 0.0,
    "survival_reward": 0.1
  },
  "danger_zone": {
    "zone_penalty": 1,
    "zone_height": 3.5
  },
  "signals": {
    "on_win_reward": 50,
    "on_knockout_reward": 10,
    "on_combo_reward": 8,
    "on_equip_reward": 15,
    "on_drop_reward": -5
  }
}
```

## Summary of Changes

**Reward Adjustments:**
- ✅ Reduced cliff penalty from massive -50 to reasonable -20
- ✅ Halved attack spam penalty from -0.04 to -0.02
- ✅ Reduced weapon drop penalty from -20 to -5
- ✅ Increased positive rewards (knockout +2, combo +3, equip +5)
- ✅ Added survival reward (+0.1/frame) for continuous positive signal

**Hyperparameter Improvements:**
- ✅ Increased exploration coefficient from 0.01 to 0.05 (5x)
- ✅ Made learning rate explicit at 3e-4
- ✅ Increased recommended training duration from 100k to 500k timesteps

**Expected Impact:**
- More balanced reward signal (positive and negative)
- Faster learning through better exploration
- More diverse agent behaviors
- Better convergence over longer training runs

These changes create a more balanced reward landscape that should result in significantly improved learning curves with rewards trending positive instead of remaining stuck in negative territory.

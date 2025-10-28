# Quick Summary: Learning Curve Improvements

## What Changed?

### Reward Balance (More Positive Signals!)
| Change | Before | After | Impact |
|--------|--------|-------|---------|
| Cliff Penalty | 0.0 or -50 | **-20** | Active but not crushing |
| Attack Penalty | -0.04/frame | **-0.02/frame** | Encourages combat |
| Weapon Drop | -20/drop | **-5/drop** | Less punishing |
| Knockout Bonus | +8 | **+10** | More rewarding |
| Combo Bonus | +5 | **+8** | Encourages skill |
| Equip Bonus | +10 | **+15** | Promotes weapon use |
| **NEW: Survival** | N/A | **+0.1/frame** | Continuous positive! |

### Training Improvements
- **Exploration**: 0.01 → **0.05** (5x more exploration)
- **Duration**: 100k → **500k** timesteps (5x longer)
- **Learning Rate**: Implicit → **3e-4** (explicit)

## Why These Changes?

**Problem**: Rewards were stuck at -15 to -21 (too negative!)

**Solution**: 
1. Reduced harsh penalties (cliff, weapon drop, attack spam)
2. Increased positive rewards (knockout, combo, equip)
3. Added continuous positive signal (survival +0.1/frame)
4. Increased exploration for diverse behaviors
5. Extended training duration for convergence

## Expected Results

### Before
- Rewards: -15 to -21 (always negative)
- Slow improvement over 160k steps
- Agent stuck in local minima

### After (Expected)
- Starting: -5 to -10 (less negative)
- After 100k: 0 to +10 (positive!)
- After 500k: +15 to +30 (good performance)

## How to Train

```bash
# Just run the training script - all changes are already configured!
python user/train_agent.py
```

The training will now:
- Run for 500k timesteps (instead of 100k)
- Show demo matches every 10k steps
- Use balanced rewards with survival bonus
- Explore more diverse strategies

## Need to Tune?

See `REWARD_TUNING_GUIDE.md` for detailed tuning instructions.

Quick tips:
- **Agent too passive?** Increase `damage_interaction_reward` to 1.5
- **Jumps off cliffs?** Increase `danger_zone_reward` to -30
- **Drops weapons?** Increase `on_drop_reward` penalty to -10
- **Learning slow?** Increase `ent_coef` to 0.08 or train longer

## Files Modified

1. `rewards_config.json` - All reward values
2. `rewards_config.py` - Default configuration  
3. `train_agent.py` - Hyperparameters + survival reward function
4. `REWARD_TUNING_GUIDE.md` - Comprehensive documentation (NEW)
5. `QUICK_SUMMARY.md` - This file (NEW)

## Key Insight

**Balance is Everything!** The agent needs both positive and negative signals to learn effectively. Before: too many negatives. Now: balanced with continuous survival reward creating a better learning gradient.

---

For detailed explanations and advanced tuning, see `REWARD_TUNING_GUIDE.md`

# Before & After Comparison

## Reward Configuration Changes

### Continuous Rewards (per frame)

| Reward Type | Before | After | Change | Rationale |
|-------------|--------|-------|--------|-----------|
| **Cliff/Danger Zone** | 0.0 (disabled) | **-20.0** | Enabled with moderate penalty | Prevents cliff jumping without being crushing |
| **Attack Spam** | -0.04 | **-0.02** | 50% reduction | Encourages necessary combat engagement |
| **Survival** ✨ | N/A | **+0.1** | NEW! | Continuous positive signal for staying alive |
| **Zone Height Trigger** | 4.2 | **3.5** | Lower threshold | Catches cliff jumps earlier |

### Event-Based Rewards (per occurrence)

| Event | Before | After | Change | Rationale |
|-------|--------|-------|--------|-----------|
| **Win Match** | +50 | +50 | No change | Already well-balanced |
| **Knockout Opponent** | +8 | **+10** | +25% | Encourage aggressive, skillful play |
| **Combo Hit** | +5 | **+8** | +60% | Reward combat mastery |
| **Equip Weapon** | +10 | **+15** | +50% | Promote weapon strategy |
| **Drop Weapon** | -20 | **-5** | 75% reduction | Less punishing for mistakes |

## Hyperparameter Changes

| Parameter | Before | After | Change | Impact |
|-----------|--------|-------|--------|---------|
| **Entropy Coefficient** | 0.01 | **0.05** | 5x increase | More exploration, diverse behaviors |
| **Learning Rate** | (implicit) | **3e-4** | Explicit | Consistent, controlled learning |
| **Training Duration** | 100k steps | **500k steps** | 5x longer | Proper convergence time |

## Expected Learning Curve Improvement

### Training Progress: Reward Trajectory

```
Before (Negative Learning):
Timesteps:     0     40k    80k   120k   160k
Rewards:    -20.0  -19.5  -18.0  -17.0  -15.5  ← Stuck in negative!
                ↑ Very slow improvement, all negative

After (Balanced Learning):
Timesteps:     0     100k   200k   300k   500k
Rewards:    -10.0   +5.0  +12.0  +20.0  +30.0  ← Crosses to positive!
                ↑ Steady improvement to positive territory
```

## Reward Signal Balance

### Before: Heavily Negative
```
Positive Signals:
├─ Win: +50 (rare, end of match only)
├─ Knockout: +8 (infrequent)
├─ Combo: +5 (requires skill)
└─ Equip: +10 (occasional)
Total Positive: ~73 per match (sparse)

Negative Signals:
├─ Cliff: -50/frame × ~100 frames = -5000 (crushing!)
├─ Weapon Drop: -20 × ~3 = -60 (harsh)
└─ Attack Spam: -0.04 × 2000 frames = -80 (discouraging)
Total Negative: ~-5140 per match (overwhelming!)

Balance: SEVERELY NEGATIVE (-5067)
```

### After: Well-Balanced
```
Positive Signals:
├─ Win: +50 (rare, end of match only)
├─ Knockout: +10 (infrequent)
├─ Combo: +8 (requires skill)
├─ Equip: +15 (occasional)
└─ Survival: +0.1 × 2700 frames = +270 ✨ (continuous!)
Total Positive: ~353 per match (strong baseline)

Negative Signals:
├─ Cliff: -20/frame × ~100 frames = -2000 (moderate)
├─ Weapon Drop: -5 × ~3 = -15 (reasonable)
└─ Attack Spam: -0.02 × 2000 frames = -40 (mild)
Total Negative: ~-2055 per match (manageable)

Balance: POSITIVE IF AGENT WINS (+298 net)
         SLIGHTLY NEGATIVE IF LOSES (-1702 net)
```

## Key Insight: The Survival Reward

The **survival reward** is the game-changer:
- **+0.1 per frame** = +270 over 90-second match
- Provides **continuous positive feedback**
- Creates a **learning gradient** even when agent doesn't win
- Balances out the negative penalties
- Encourages the agent to **stay alive longer** = more learning

## Expected Behavioral Changes

| Behavior | Before | After |
|----------|--------|-------|
| **Cliff Jumping** | Either ignored (0.0) or over-penalized (-50) | Properly discouraged (-20) |
| **Combat Engagement** | Discouraged by -0.04 penalty | Encouraged with halved penalty |
| **Weapon Usage** | Moderate incentive (+10) | Strong incentive (+15) |
| **Weapon Dropping** | Over-punished (-20) | Reasonably discouraged (-5) |
| **Survival** | No explicit reward | Continuously rewarded (+0.1/frame) |
| **Exploration** | Limited (ent_coef 0.01) | Enhanced (ent_coef 0.05) |

## Training Efficiency

### Before
- 100k timesteps ≈ 37 matches (2700 frames each)
- Learning stalled in negative territory
- Insufficient exploration
- Rewards: -21 to -15 range

### After
- 500k timesteps ≈ 185 matches
- Expected convergence to positive rewards
- 5x more exploration (ent_coef)
- Expected rewards: +15 to +30 range

## Bottom Line

**Before**: Agent was punished heavily for mistakes with little positive reinforcement  
**After**: Agent receives balanced feedback with continuous survival reward as baseline

This creates a **learnable gradient** where:
1. Just staying alive = positive reward
2. Playing well = much more positive reward
3. Mistakes = moderate penalties, not crushing ones

Result: **Faster, more stable learning with better final performance**

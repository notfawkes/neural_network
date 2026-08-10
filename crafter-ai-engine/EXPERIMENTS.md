# Empirical Experiments & Benchmark Results

This document describes the experimental methodology and evaluation benchmarks on Crafter.

---

## Benchmark Metrics

Crafter evaluates agents across a standardized budget of 1,000,000 environment steps using:

1. **22 Semantic Achievements**: Percentage of episodes where an achievement was successfully unlocked.
2. **Geometric Mean Crafter Score**:
   $$\text{Score} = \exp\left( \frac{1}{22} \sum_{i=1}^{22} \ln(s_i + 10^{-6}) \right)$$
   where $s_i \in [0, 1]$ is the empirical success rate of achievement $i$. Improvements on rare milestones (e.g. `defeat_skeleton`, `make_stone_pickaxe`) contribute exponentially more than common tasks.
3. **Episode Return**: Reward accumulated through achievement discoveries (+1.0) and health regeneration (+0.1 / -0.1).
4. **Survival Duration**: Average timesteps survived before starvation or enemy defeat.

---

## Progression Milestones

### Phase 1: Early Random Exploration (0 - 50k steps)
- **Behaviors**: Erratic wandering, frequent sleep attempts during daytime, starvation.
- **Achievements Unlocked**: `wake_up` (100%), occasional `eat_plant`.
- **Mean Reward**: -0.8 to -0.2.
- **Average Length**: 40 - 80 steps.

### Phase 2: Survival & Foraging Discovery (50k - 200k steps)
- **Behaviors**: Learns to avoid immediate water drownings, consumes cows and plants when health declines.
- **Achievements Unlocked**: `collect_wood`, `eat_cow`, `eat_plant`, `collect_drink`.
- **Mean Reward**: +0.5 to +2.1.
- **Average Length**: 120 - 250 steps.

### Phase 3: Tool Crafting & Combat Initiation (200k - 1M steps)
- **Behaviors**: Constructs crafting tables, crafts wood and stone pickaxes, navigates night cycles without immediate death.
- **Achievements Unlocked**: `place_table`, `make_wood_pickaxe`, `collect_stone`, `make_stone_pickaxe`, `defeat_zombie`.
- **Mean Reward**: +3.5 to +8.2.
- **Average Length**: 350 - 700+ steps.

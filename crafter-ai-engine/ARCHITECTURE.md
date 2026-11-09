# Engine Architecture Specification

This document details the internal design of `crafter-ai-engine`, covering tensor representations, neural forward pipelines, temporal memory propagation, and vectorized environment handling.

---

## 1. Perception Pipeline: `CrafterCNN`

The visual perception module processes raw 64×64 RGB observations without downsampling or pooling layers to preserve small game entities (arrows, saplings, cows, zombies).

```
Input Frame: [B, 3, 64, 64] float32 normalized ∈ [0.0, 1.0]
    │
    ▼
Conv2D (Filters: 32, Kernel: 3, Stride: 2, Padding: 0)  ──► [B, 32, 31, 31]
ReLU
    │
    ▼
Conv2D (Filters: 64, Kernel: 3, Stride: 2, Padding: 0)  ──► [B, 64, 15, 15]
ReLU
    │
    ▼
Conv2D (Filters: 64, Kernel: 3, Stride: 2, Padding: 0)  ──► [B, 64, 7, 7]
ReLU
    │
    ▼
Flatten (64 × 7 × 7)                                     ──► [B, 3136]
    │
    ▼
Linear (3136 → 512)                                      ──► [B, 512]
ReLU
```

- **Orthogonal Initialization**: Weights are initialized with orthogonal matrices with gain $\sqrt{2}$ for ReLU activations, preventing exploding/vanishing gradients early in training.
- **5D Tensor Support**: Supports both standard batch inputs $[B, C, H, W]$ and sequential trajectory tensors $[T, B, C, H, W]$.

---

## 2. Recurrent Memory: `RecurrentActorCritic`

Crafter is a partially observable Markov decision process (POMDP): player inventory, monster pathing, daylight cycles, and craftable recipes require history context.

```
Visual Features: [T, B, 512]
    │
    ▼
LSTM Layer (Input: 512, Hidden: 256, Layers: 1)
    ├── Hidden State h_t: [1, B, 256]
    └── Cell State   c_t: [1, B, 256]
    │
    ├── Reset Mask (1.0 - done_{t-1}) ensures state is zeroed on episode resets
    ▼
Recurrent Representation: [T, B, 256]
    │
    ├──► Actor Linear (256 → 17)   ──► Action Logits ──► Categorical Distribution
    └──► Critic Linear (256 → 1)   ──► State-Value Estimate V(s)
```

### Episode Boundary Handling
During vectorized rollout collection across $B$ parallel environments, episodes terminate asynchronously. At step $t$, if environment $b$ reached terminal status at $t-1$, its hidden and cell state vectors are masked before feeding the step $t$ visual feature:

$$h_{t-1}^{(b)} \leftarrow h_{t-1}^{(b)} \cdot (1 - \text{done}_{t-1}^{(b)})$$
$$c_{t-1}^{(b)} \leftarrow c_{t-1}^{(b)} \cdot (1 - \text{done}_{t-1}^{(b)})$$

---

## 3. Rollout Buffer & Trajectory Sequencing

Unlike standard feedforward PPO implementations which shuffle transitions across the entire $[T \times B]$ batch, `crafter-ai-engine` maintains sequence continuity:

- Transitions are grouped by environment sequence: $[T, B, \dots]$.
- Mini-batches preserve the temporal order $0, 1, \dots, T-1$.
- Initial recurrent states $(h_0, c_0)$ are captured at the beginning of rollout collection and passed to the optimizer during update epochs.


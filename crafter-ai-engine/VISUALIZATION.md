# Visual Learning Observatory & Interpretability Architecture

This document describes how CRAfter AI provides transparent interpretability into deep RL models.

---

## 1. Pixel-Preserving Observation Scaling

Neural visual observations are strictly $3 \times 64 \times 64$ float32 tensors. When displayed to humans, linear interpolation (bilinear or bicubic) creates blurred artifacts that obscure game tiles and items.

The visualizer enforces:
```css
image-rendering: pixelated;
image-rendering: -moz-crisp-edges;
image-rendering: crisp-edges;
```
Magnified $5\times$ to $320 \times 320$ pixels, individual pixels remain sharp and faithful to what the convolutional filters receive.

---

## 2. Real-Time Action Probability Stream

At each inference step, the actor outputs raw logits $z \in \mathbb{R}^{17}$. The visual observatory displays:
- Softmax probability bars $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ across all 17 actions.
- The sampled action during training vs argmax greedy action during evaluation.
- Instantaneous entropy $-\sum p_i \ln p_i$, indicating agent confidence or confusion.

---

## 3. Failure & Mortality Timeline

When an episode concludes with `terminated = True` (health reaches 0), the episode recorder preserves:
1. The last 10 frames prior to fatal injury.
2. The specific actions taken under enemy attack.
3. Critic state-value drops indicating anticipation of death.
4. Trajectory coordinates showing where in the world the agent perished.

---

## 4. Counterfactual Checkpoint Comparison

To verify that neural updates result in genuine policy improvements rather than random walk differences:
1. Two model checkpoints (e.g. step 50k and step 250k) are loaded into memory.
2. Identical observation frames are presented to both models.
3. The observatory plots the comparative shifts in action distributions:
   - For example: when facing a zombie, Checkpoint A displays 12% attack probability, whereas Checkpoint B displays 68% attack probability.

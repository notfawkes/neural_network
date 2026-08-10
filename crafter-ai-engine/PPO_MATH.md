# Mathematical Formulation of Custom PPO

This document specifies the exact equations implemented in `crafter_ai.training.ppo` and `crafter_ai.training.ppo_recurrent`.

---

## 1. Generalized Advantage Estimation (GAE)

Given discounted return factor $\gamma = 0.99$ and GAE factor $\lambda = 0.95$, temporal-difference errors $\delta_t$ are computed backwards from step $T-1$ down to $0$:

$$\delta_t = r_t + \gamma V(s_{t+1}) (1 - d_t) - V(s_t)$$

where $d_t \in \{0, 1\}$ is the episode termination indicator.

The recursive advantage estimator $\hat{A}_t$ is:

$$\hat{A}_t = \delta_t + (\gamma \lambda) (1 - d_t) \hat{A}_{t+1}$$

Targets for critic optimization (returns) are:

$$\hat{R}_t = \hat{A}_t + V(s_t)$$

Advantages are normalized across the batch before gradient updates:

$$\hat{A}_t^{\text{norm}} = \frac{\hat{A}_t - \mu(\hat{A})}{\sigma(\hat{A}) + 10^{-8}}$$

---

## 2. Clipped Surrogate Policy Objective

For policy parameters $\theta$ and old collection policy $\theta_{\text{old}}$, the probability ratio $r_t(\theta)$ is:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} = \exp\left( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t) \right)$$

The clipped surrogate loss $\mathcal{L}^{\text{CLIP}}(\theta)$ is:

$$\mathcal{L}^{\text{CLIP}}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

with clipping parameter $\epsilon = 0.2$.

---

## 3. Value Function & Entropy Loss

The critic value loss is computed as squared error against empirical returns:

$$\mathcal{L}^{\text{VF}}(\theta) = \frac{1}{2} \mathbb{E}_t \left[ \left( V_\theta(s_t) - \hat{R}_t \right)^2 \right]$$

The entropy regularization term for exploration is:

$$\mathcal{H}(\pi_\theta(\cdot \mid s_t)) = -\sum_{a=0}^{16} \pi_\theta(a \mid s_t) \log \pi_\theta(a \mid s_t)$$

$$\mathcal{L}^{\text{ENT}}(\theta) = -c_2 \, \mathbb{E}_t \left[ \mathcal{H}(\pi_\theta(\cdot \mid s_t)) \right]$$

with entropy coefficient $c_2 = 0.01$.

---

## 4. Total Composite Objective

The neural parameters are updated to minimize:

$$\mathcal{L}^{\text{TOTAL}}(\theta) = \mathcal{L}^{\text{CLIP}}(\theta) + c_1 \mathcal{L}^{\text{VF}}(\theta) + \mathcal{L}^{\text{ENT}}(\theta)$$

where $c_1 = 0.5$. Gradients are clipped to a maximum Euclidean norm of $0.5$:

$$g \leftarrow g \cdot \min\left(1, \, \frac{0.5}{\|g\|_2 + 10^{-6}}\right)$$

---

## 5. Diagnostic Metrics

- **Approximate KL Divergence**:
  $$D_{\text{KL}}^{\text{approx}} = \mathbb{E}_t \left[ (r_t - 1) - \log r_t \right]$$
- **Clip Fraction**: Percentage of transitions where $|r_t - 1| > \epsilon$.

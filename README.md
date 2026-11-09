<div align="center">

# 🎮 CRAfter AI — Visual Reinforcement Learning Lab

### Autonomous Neural Agent Learning Open-World Survival from Raw Pixels

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14_App_Router-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

[**Live POV Observatory**](#-1-live-agent-observatory) • [**Interactive Dashboard**](#-3-dashboard-features) • [**Empirical Figures**](#-empirical-training-figures--findings) • [**Mathematical Formulations**](#-mathematical-foundations) • [**Quickstart**](#-quickstart--execution)

</div>

---

## 📖 Overview

**CRAfter AI** is a visual deep reinforcement learning research laboratory designed to investigate how artificial agents discover survival mechanics, gather materials, craft tools, and navigate hostile environments entirely from **raw $64 \times 64$ RGB pixel observations**.

Unlike rule-based game bots or scripted state machines, this neural agent:
- Receives **no symbolic coordinates, inventory objects, or world coordinates**.
- Interacts exclusively through an egocentric $64 \times 64 \times 3$ image stream and 17 discrete motor actions.
- Learns from trial-and-error by optimizing a **from-scratch PyTorch implementation** of Recurrent Proximal Policy Optimization (**CNN $\to$ LSTM $\to$ Actor-Critic PPO**).
- Discovers the 22-milestone Crafter crafting tech tree through exploratory policy updates.

The core philosophy of this project is **radical visual transparency**: you can watch the agent play in real-time, inspect internal CNN visual representations and LSTM recurrent memories, scrub through fatal decisions on an interactive timeline, and compare baseline vs trained checkpoint behaviors side-by-side.

---

## 🏛️ System Architecture

The laboratory is organized into three decoupled repositories:

```mermaid
graph LR
    subgraph Engine ["crafter-ai-engine (Pure PyTorch)"]
        Env["Crafter Vectorized Envs (4x)"] --> CNN["3-Layer ConvNet (512-d)"]
        CNN --> LSTM["Recurrent Core (LSTM 256-d)"]
        LSTM --> Actor["Actor Head (17 Logits)"]
        LSTM --> Critic["Critic Head (V(s))"]
        Actor & Critic --> PPO["Custom PPO + GAE-λ Trainer"]
        PPO --> Rec["Threaded Async Episode Recorder"]
        PPO --> Storage["Atomic IPC State & Live Frame Store"]
    end

    subgraph Backend ["crafter-ai-backend (FastAPI)"]
        Storage --> WS["WebSocket Streaming (/ws/live-game, /ws/training)"]
        Rec --> EpService["Episode & Frame Replay Service"]
        Storage --> MetService["Metrics & Telemetry Service"]
        PPO --> ModService["Counterfactual Policy Compare Service"]
    end

    subgraph Frontend ["crafter-ai-dashboard (Next.js 14)"]
        WS --> LiveView["Live POV & PPO Cycle Observatory (/live)"]
        EpService --> EpExp["Episode Explorer & Failure Scrubber (/episodes)"]
        MetService --> AnaView["Learning Analysis & Figure Gallery (/analysis)"]
        ModService --> ExpView["Checkpoint Policy Comparator (/experiments)"]
    end
```

### 1. `crafter-ai-engine` (Deep RL Core)
- **Zero Prebuilt RL Frameworks**: Built from first principles in PyTorch without Stable-Baselines3, Ray RLlib, or CleanRL.
- **Visual Encoder**: 3 Conv2D layers with stride 2 and orthogonal weight initializations, projecting raw visual inputs into a 512-dimensional visual embedding vector.
- **Recurrent Working Memory**: 1-layer LSTM (hidden size 256) maintaining spatial-temporal memory across timesteps, correctly masked and reset across episode boundaries.
- **Actor-Critic Heads**: Dual linear heads outputting a 17-action Categorical distribution and a scalar state-value estimate $V(s)$.
- **Rollout Buffer & GAE**: Vectorized rollout buffer storing multi-worker transitions, computing Generalized Advantage Estimation ($\text{GAE-}\lambda$).
- **Scientific Visualization Engine**: Automated 10-figure generator outputting 300 DPI research figures.

### 2. `crafter-ai-backend` (High-Performance Server)
- **FastAPI Async Server**: Non-blocking asynchronous service managing training subprocesses with safe signal termination (`SIGINT`/`SIGTERM`).
- **Live WebSocket Streams**: Bounded 10–15 FPS frame streaming with nearest-neighbour scaling to ensure zero training loop blockage.
- **Episode & Replay Service**: Reads compressed `.npz` episode archives and serves frame-by-frame base64 images and step telemetry.
- **Policy Comparison Service**: Evaluates two checkpoints side-by-side on real environment observations counterfactually.
- **Figures API**: Direct streaming and on-demand rendering of high-resolution scientific evaluation plots.

### 3. `crafter-ai-dashboard` (Research UI)
- **Next.js 14 App Router**: TypeScript, Tailwind CSS, dark-themed Linear/Vercel research interface.
- **Pixel-Crisp Canvas**: Nearest-neighbour scaling (`image-rendering: pixelated`) of raw $64 \times 64$ inputs.
- **Real-Time Telemetry**: Real-time charts of returns, policy entropy, value loss, and action probability distributions.
- **Failure Timeline Scrubber**: Step-by-step scrubbing with instant "Jump to Damage" and "Jump to Death" debugging buttons.

---

## 📊 Empirical Training Figures & Findings

All figures below are automatically generated and saved directly to the [`figures/`](figures) folder upon completion of training or on-demand via `python scripts/plot_figures.py`.

### Figure 1: Reinforcement Learning Dynamics
![Learning Curves](figures/01_learning_curves.png)
> **Figure 1**: *(Left)* Episode undiscounted cumulative return across training episodes showing the raw returns (light points) and the rolling average with a $\pm 1\sigma$ standard deviation shadow band. *(Right)* Agent survival duration in steps per episode, demonstrating dramatic survival expansion as policy discovers how to avoid starvation and evade night-time monsters.

---

### Figure 2: PPO Optimization Loss Diagnostics
![PPO Losses](figures/02_ppo_training_losses.png)
> **Figure 2**: Four-panel PPO convergence telemetry across 1,000,000 environment steps. *(Top-Left)* Clipped surrogate policy loss. *(Top-Right)* Critic mean squared value estimation error $\mathbb{E}[(V_\theta(s) - R_t)^2]$. *(Bottom-Left)* Policy entropy $H(\pi)$ showing steady, controlled decay from uniform exploration to decisive action selection. *(Bottom-Right)* Approximate KL divergence bounded below the target threshold $\delta=0.015$.

---

### Figure 3: Crafter Benchmark Score Progression
![Crafter Score](figures/03_crafter_score_progression.png)
> **Figure 3**: Official Crafter benchmark geometric mean score (\%) calculated as $S = \exp\left(\frac{1}{22}\sum_{i=1}^{22} \ln(1 + s_i)\right) - 1$. The neural agent successfully surpasses the standard Random Policy baseline ($\sim 1.6\%$) and matches standard benchmark baselines.

---

### Figure 4: 22-Milestone Achievement Unlock Profile
![Achievement Rates](figures/04_achievement_success_rates.png)
> **Figure 4**: Comprehensive breakdown of all 22 Crafter achievements categorized into 7 domains: **Survival** (green), **Gathering** (lime), **Mining** (cyan), **Crafting** (blue), **Building** (purple), **Combat** (rose), and **Farming** (amber). The agent consistently masters early survival and wood crafting before expanding into stone tool production.

---

### Figure 5: Achievement Discovery & Mastery Heatmap
![Achievement Heatmap](figures/05_achievement_heatmap.png)
> **Figure 5**: Temporal heatmap showing the discovery window of specific game milestones across training time (Window 1: early exploration $\to$ Window 10: mature policy). Note the progression from basic sustenance (`eat_plant`, `collect_wood`) to complex multi-tier dependencies (`make_wood_pickaxe`, `collect_stone`, `make_stone_pickaxe`).

---

### Figure 6: Policy Discrete Action Distribution
![Action Policy Distribution](figures/06_action_policy_distribution.png)
> **Figure 6**: Selection frequencies across the 17 discrete Crafter actions. The trained policy transitions away from redundant actions toward high-utility survival actions: active interaction (`do`), spatial traversal (`move_*`), crafting (`make_*`), and construction (`place_*`).

---

### Figure 7: Spatial World Exploration Heatmap
![Spatial Exploration Heatmap](figures/07_spatial_exploration_heatmap.png)
> **Figure 7**: 2D trajectory density heatmap of agent $(X, Y)$ world coordinates across recorded episodes. Starting from the central spawn point $[32, 32]$ (marked by the red star), the agent progressively learns to explore outwards into distinct biomes (grassland, forest, mountains, lakes) in search of coal, iron, and cows.

---

### Figure 8: Neural Internal Representations & Working Memory
![Neural Representations](figures/08_neural_representation_inspection.png)
> **Figure 8**: Inspection of internal neural network activity for a single step. *(Left)* Raw visual observation input ($64 \times 64 \times 3$). *(Center)* 512-dimensional CNN visual feature vector activations highlighting dominant filter responses. *(Right)* 256-dimensional LSTM recurrent hidden state $\mathbf{h}_t$, illustrating how memory of past events (e.g. nightfall or nearby monsters) persists across frames.

---

### Figure 9: Policy Evolution Across Model Checkpoints
![Model Checkpoint Comparison](figures/09_model_checkpoint_comparison.png)
> **Figure 9**: Comparative performance across four model checkpoint stages (`Initial Policy`, `Step 50k`, `Step 100k`, and `Final Policy`) showing cumulative return progression, mean survival timesteps, and total count of unique Crafter achievements unlocked.

---

### Figure 10: Master Training Overview Dashboard
![Training Summary Dashboard](figures/10_training_summary_dashboard.png)
> **Figure 10**: High-level multi-panel research summary synthesizing episode returns, geometric Crafter score gauge, critic MSE convergence, policy exploration entropy, and the top discovered achievements in a publication-grade dark scientific theme.

---

## 🔭 The Live Agent Observatory

### 1. Live Agent POV
The live observatory displays the real-time perspective of the agent as it interacts with the Crafter environment:
- **Exact Pixel Rendering**: Scaled with nearest-neighbour interpolation to highlight raw pixel observations.
- **Playback Controls**: Play, Pause, Step Forward (single-step inspection mode), Replay, and variable speed toggles (`0.5x`, `1x`, `2x`, `5x`).
- **Telemetry HUD**: Active episode, step counter, health status ($H / 9$ HP), world coordinates $[X, Y]$, discrete action, and step reward $\Delta r$.

### 2. PPO Training Cycle Visualizer
Explicitly visualizes the 5-stage recurrent PPO loop:
$$\text{COLLECT ROLLOUT (Step X/128)} \longrightarrow \text{STORE BUFFER} \longrightarrow \text{CALCULATE GAE-λ} \longrightarrow \text{PPO UPDATE (Epoch X/4)} \longrightarrow \text{UPDATED POLICY}$$
- **Educational Context**: Explains that PPO updates in rollout batches across parallel workers, rather than reacting instantly to isolated deaths.

### 3. Episode Memory Explorer & Failure Scrubber
- **Interactive Scrubber**: Move through any recorded episode frame-by-frame.
- **Failure Pinpointing**: Instant `Jump to Damage` (first negative reward step) and `Jump to Death` (fatal step) buttons.
- **Policy Replay**: Displays the exact 17-action probability distribution and critic estimate $V(s)$ at each historical step.

### 4. Side-by-Side Episode Learning Comparison
- Replay an early baseline run (Episode 1) alongside a later policy run (Episode 2+) synchronously to observe behavioral divergence.

---

## 📐 Mathematical Foundations

### 1. Convolutional Feature Extractor
Given an input observation $\mathbf{o}_t \in \mathbb{R}^{3 \times 64 \times 64}$ with pixel values normalized to $[0, 1]$:
$$\mathbf{z}_t = \text{MLP}(\text{Flatten}(\text{Conv}_3(\text{ReLU}(\text{Conv}_2(\text{ReLU}(\text{Conv}_1(\mathbf{o}_t))))))) \in \mathbb{R}^{512}$$

### 2. Recurrent Working Memory (LSTM Core)
To handle partial observability (day-night cycles, hidden monsters, internal hunger/thirst timers):
$$\mathbf{h}_t, \mathbf{c}_t = \text{LSTM}(\mathbf{z}_t, (1 - d_{t-1})\mathbf{h}_{t-1}, (1 - d_{t-1})\mathbf{c}_{t-1})$$
where $d_{t-1} \in \{0, 1\}$ is the episode termination flag that resets recurrent states at episode boundaries.

### 3. Actor & Critic Heads
$$\pi_\theta(a \mid s_t) = \text{Softmax}(\mathbf{W}_\pi \mathbf{h}_t + \mathbf{b}_\pi) \in \Delta^{17}, \quad V_\phi(s_t) = \mathbf{W}_v \mathbf{h}_t + b_v \in \mathbb{R}$$

### 4. Generalized Advantage Estimation (GAE-λ)
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V, \quad \text{where } \delta_t^V = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$$
With hyperparameters $\gamma = 0.99$ and $\lambda = 0.95$.

### 5. Clipped Surrogate PPO Objective
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta) \hat{A}_t, \, \text{clip}(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$
where $\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ and clipping threshold $\epsilon = 0.2$.

Total loss optimized with gradient clipping $\|\nabla_\theta L\| \le 0.5$:
$$L^{\text{TOTAL}}(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_1 \cdot \frac{1}{2} \hat{\mathbb{E}}_t \left[ (V_\phi(s_t) - \hat{R}_t)^2 \right] - c_2 \cdot \hat{\mathbb{E}}_t \left[ H(\pi_\theta(\cdot \mid s_t)) \right]$$
with $c_1 = 0.5$, $c_2 = 0.01$.

---

## 🎯 Action Space & Achievement Matrix

### The 17 Discrete Actions
| ID | Action Name | Category | Description |
|:---:|:---|:---|:---|
| `0` | `noop` | Idle | Do nothing for 1 timestep |
| `1` | `move_left` | Movement | Move agent left 1 tile |
| `2` | `move_right` | Movement | Move agent right 1 tile |
| `3` | `move_up` | Movement | Move agent up 1 tile |
| `4` | `move_down` | Movement | Move agent down 1 tile |
| `5` | `do` | Interaction | Hit, collect, attack, or mine block in facing direction |
| `6` | `sleep` | Survival | Sleep to regenerate health and pass night time |
| `7` | `place_stone` | Construction | Place stone obstacle/wall |
| `8` | `place_table` | Construction | Place crafting table (enables tool crafting) |
| `9` | `place_furnace`| Construction | Place smelting furnace (enables iron smelting) |
| `10`| `place_plant` | Farming | Plant a sapling for food production |
| `11`| `make_wood_pickaxe` | Crafting | Craft wooden pickaxe (requires table + wood) |
| `12`| `make_stone_pickaxe`| Crafting | Craft stone pickaxe (requires table + stone) |
| `13`| `make_iron_pickaxe` | Crafting | Craft iron pickaxe (requires table + furnace + iron) |
| `14`| `make_wood_sword` | Combat | Craft wooden sword for self-defense |
| `15`| `make_stone_sword`| Combat | Craft stone sword |
| `16`| `make_iron_sword` | Combat | Craft iron sword |

### The 22 Crafter Achievements
- **Survival**: `wake_up`, `eat_plant`, `eat_cow`, `collect_drink`
- **Gathering**: `collect_wood`, `collect_sapling`
- **Mining**: `collect_stone`, `collect_coal`, `collect_iron`, `collect_diamond`
- **Building**: `place_table`, `place_furnace`, `place_stone`
- **Farming**: `place_plant`
- **Crafting**: `make_wood_pickaxe`, `make_stone_pickaxe`, `make_iron_pickaxe`, `make_wood_sword`, `make_stone_sword`, `make_iron_sword`
- **Combat**: `defeat_zombie`, `defeat_skeleton`

---

## 🚀 Quickstart & Execution

### 1. Prerequisites & Virtual Environment
Ensure you have Python 3.10+ (Python 3.12 recommended) and Node.js 18+ installed.

```bash
# Clone the repository
git clone https://github.com/balasudalaimuthu/crafter.git
cd crafter

# Activate or create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install master dependencies and editable packages
pip install -r requirements.txt
pip install -e crafter-game
pip install -e crafter-ai-engine
```

### 2. Verify with Test Suites
```bash
# Run all 66 engine unit & integration tests
cd crafter-ai-engine
pytest tests/ -v

# Run all 9 backend API & WebSocket tests
cd ../crafter-ai-backend
pytest tests/ -v
```

### 3. Launch the Complete Lab (Single Command)
Use the included controller script to launch all services simultaneously:

```bash
./start_lab.sh
```
- **FastAPI Backend Server**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Next.js Research Dashboard**: [http://localhost:3000](http://localhost:3000)

Alternatively, launch services in dedicated terminals:
```bash
# Terminal 1: Backend
./start_lab.sh backend

# Terminal 2: Dashboard
./start_lab.sh frontend
```

### 4. Train the Neural Agent
To start a new training experiment or resume from a checkpoint:
```bash
# Run training with default CNN + LSTM PPO config (Apple Silicon MPS / CUDA / CPU)
python crafter-ai-engine/scripts/train.py --config crafter-ai-engine/configs/default.yaml

# Train for a specific timestep budget (e.g. 50,000 steps)
python crafter-ai-engine/scripts/train.py --config crafter-ai-engine/configs/default.yaml --total-timesteps 50000

# Resume training from an existing checkpoint
python crafter-ai-engine/scripts/train.py --config crafter-ai-engine/configs/default.yaml --resume checkpoints/crafter_lstm_ppo_latest.pt
```

### 5. Generate Figures on Demand
```bash
python crafter-ai-engine/scripts/plot_figures.py --output-dir figures --experiment crafter_lstm_ppo
```

---

## 📂 Repository Structure

```
crafter/
├── figures/                              # 10 publication-grade scientific figures
│   ├── 01_learning_curves.png
│   ├── 02_ppo_training_losses.png
│   ├── 03_crafter_score_progression.png
│   ├── 04_achievement_success_rates.png
│   ├── 05_achievement_heatmap.png
│   ├── 06_action_policy_distribution.png
│   ├── 07_spatial_exploration_heatmap.png
│   ├── 08_neural_representation_inspection.png
│   ├── 09_model_checkpoint_comparison.png
│   └── 10_training_summary_dashboard.png
├── checkpoints/                          # Saved model weights (.pt) & index metadata
├── recordings/                           # Compressed .npz episode trajectory archives
├── logs/                                 # Training metrics JSON logs & run transcripts
├── crafter-game/                         # Localized Crafter game package (editable)
├── crafter-ai-engine/                    # Deep RL PyTorch Engine
│   ├── crafter_ai/
│   │   ├── env/                          # Gym adapter, vectorized environments, frame stack
│   │   ├── models/                       # CrafterCNN, LSTM, RecurrentActorCritic
│   │   ├── training/                     # RolloutBuffer, PPO, Recurrent PPO with GAE
│   │   ├── recording/                    # Async threaded EpisodeRecorder & shared storage
│   │   ├── evaluation/                   # Evaluator & 10-figure scientific generator
│   │   └── utils/                        # CheckpointManager, MetricsTracker, MPS device tools
│   ├── configs/                          # Experiment configs (default.yaml, baseline.yaml)
│   ├── scripts/                          # train.py, evaluate.py, plot_figures.py, play.py
│   └── tests/                            # Unit & integration test suite (66 tests)
├── crafter-ai-backend/                   # FastAPI Server & Streaming
│   ├── api/
│   │   ├── routers/                      # training, episodes, models, figures, websocket
│   │   └── schemas/                      # Pydantic v2 schemas
│   ├── core/                             # Config & ProcessManager (subprocess runner)
│   ├── services/                         # EpisodeService, ModelService, MetricsService
│   └── tests/                            # REST & WebSocket endpoint tests (9 tests)
├── crafter-ai-dashboard/                 # Next.js 14 Research Observatory
│   ├── src/
│   │   ├── app/                          # App Router pages (/live, /episodes, /analysis, etc.)
│   │   ├── components/                   # GameView, ActionProbChart, EpisodeReplay, etc.
│   │   ├── lib/                          # API client & formatting helpers
│   │   └── types/                        # TypeScript interfaces
├── requirements.txt                      # Unified master dependency list
├── start_lab.sh                          # Single-command lab control script
├── LICENSE                               # MIT License
└── README.md                             # Research documentation & visual guide
```

---

## 🔬 Benchmark Hardware & Performance

Benchmarked on **Apple Silicon M4 (macOS 15, MPS backend)**:
- **Environment Throughput**: $\sim 1,694 \text{ steps/sec}$
- **Rollout Collection (4 workers)**: $\sim 350 \text{ FPS}$
- **Full PPO Training Throughput**: $\sim 125\text{--}180 \text{ FPS}$ (including gradient updates and asynchronous compressed recording)
- **Memory Footprint**: $< 1.2 \text{ GB RAM}$

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). The Crafter environment is created by Danijar Hafner and licensed under the MIT License.

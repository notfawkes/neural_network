# CRAfter AI — Visual Reinforcement Learning Lab

An interactive deep reinforcement learning laboratory and benchmark suite for training neural agents to master the open-world survival game **Crafter** directly from raw 64×64 RGB visual observations.

Unlike conventional game bots that rely on handcrafted heuristics or privileged engine states, CRAfter AI implements an autonomous neural agent powered by **Convolutional Neural Networks (CNN)**, **Long Short-Term Memory (LSTM)** recurrent memory, and an from-scratch implementation of **Proximal Policy Optimization (PPO)** with Generalized Advantage Estimation (GAE).

The core philosophy of this project is **interpretability and observable learning**: researchers and engineers can visually track the agent's exact field of view, observe its moment-to-moment decisions and action probability distributions, analyze trajectory exploration trails, inspect death sequences, and perform counterfactual policy comparisons between model checkpoints.

---

## System Architecture

The project consists of three dedicated components:

1. **`crafter-ai-engine`**: Pure PyTorch machine learning engine containing the custom environment adapter, CNN-LSTM actor-critic models, rollout buffer, vectorized environment runner, and custom PPO optimizer.
2. **`crafter-ai-backend`**: High-performance FastAPI backend server managing training subprocesses, streaming live 64×64 agent observations over WebSockets, serving compressed episode replays, and providing checkpoint evaluation APIs.
3. **`crafter-ai-dashboard`**: Dark-themed Next.js 14 research interface with real-time observation playback, 17-action probability visualization, failure timeline inspectors, milestone episode comparisons, and achievement trackers.
4. **`crafter-game`**: The underlying Crafter open-world environment packaged cleanly as a localized submodule.

---

## Key Features

- **End-to-End Visual Learning**: Agents receive solely 3×64×64 normalized RGB frames with 17 discrete actions.
- **Custom PPO Implementation**: Full PyTorch implementation without blackbox libraries (no Stable-Baselines3 or RLlib).
- **Recurrent Sequence Handling**: Proper sequential LSTM unrolling across episode boundaries; zero-state resets on terminal steps.
- **Nearest-Neighbor Visualizer**: 64×64 visual observations magnified without bilinear smoothing, preserving raw pixel integrity.
- **Counterfactual Policy Comparison**: Evaluates two checkpoints on identical visual scenes to measure shifts in action distributions.
- **Failure Timeline Analysis**: Chronological replay and decision inspection leading up to agent mortality.
- **Apple Silicon (M4/MPS) Acceleration**: Out-of-the-box support for PyTorch MPS and CPU execution.

---

## Directory Overview

```
crafter/
├── crafter-ai-engine/           # Core RL PyTorch Engine
│   ├── configs/                 # YAML training configurations
│   ├── crafter_ai/              # Python package
│   │   ├── env/                 # Gym/Gymnasium adapter & wrappers
│   │   ├── models/              # CrafterCNN, ActorCritic, RecurrentActorCritic
│   │   ├── training/            # Custom RolloutBuffer, PPO, Recurrent PPO
│   │   ├── recording/           # Bounded async episode recorder
│   │   ├── evaluation/          # 22-Achievement Evaluator & Crafter Score
│   │   └── utils/               # CheckpointManager, Device benchmark, Metrics
│   ├── scripts/                 # CLI: train.py, evaluate.py, play.py, compare.py
│   └── tests/                   # Pytest suite (unit & integration)
├── crafter-ai-backend/          # FastAPI Management & Streaming Server
│   ├── api/                     # REST routers & WebSockets
│   ├── core/                    # ProcessManager & Settings
│   ├── services/                # Episode, Metrics & Model services
│   └── tests/                   # API test suite
├── crafter-ai-dashboard/        # Next.js 14 Research Dashboard
│   ├── src/app/                 # App Router (7 pages)
│   ├── src/components/          # Live POV, Replay, Charts & Diagrams
│   └── src/lib/                 # API client & helpers
└── crafter-game/                # Packaged Crafter environment
```

---

## Quickstart

### 1. Run Unit & Integration Tests

```bash
cd crafter-ai-engine
python3 -m pytest tests/ -v
```

### 2. Inspect Environment & Benchmark Hardware

```bash
cd crafter-ai-engine
python3 scripts/inspect_environment.py
```

### 3. Start Training via CLI

```bash
cd crafter-ai-engine
python3 scripts/train.py --config configs/default.yaml
```

### 4. Launch Backend Server

```bash
cd crafter-ai-backend
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Launch Visual Dashboard

```bash
cd crafter-ai-dashboard
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the live dashboard.

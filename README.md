# CRAfter AI — Visual Reinforcement Learning Lab

A visual deep reinforcement learning research lab and benchmark suite for training neural agents to master the open-world survival game **Crafter** from raw 64×64 visual observations.

---

## Workspace Structure

- **`crafter-ai-engine/`**: Pure PyTorch machine learning library featuring custom environment adapters, 3-layer CNN feature extractors, recurrent LSTM memory, and custom Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE).
- **`crafter-ai-backend/`**: FastAPI backend server supporting training process management, live observation streaming over WebSockets, compressed episode replay serving, and checkpoint comparison APIs.
- **`crafter-ai-dashboard/`**: Dark-themed research visualizer built with Next.js 14 and Tailwind CSS, featuring pixelated observation displays, 17-action categorical distribution bars, milestone comparisons, and achievement trackers.
- **`crafter-game/`**: The localized Crafter open-world game environment.

---

## Quickstart

### 1. Verify Environment & Benchmark Hardware
```bash
cd crafter-ai-engine
python3 scripts/inspect_environment.py
```

### 2. Run Comprehensive Test Suite
```bash
# Engine tests (unit & integration)
cd crafter-ai-engine
python3 -m pytest tests/ -v

# Backend tests
cd ../crafter-ai-backend
python3 -m pytest tests/ -v
```

### 3. Launch Backend
```bash
cd crafter-ai-backend
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Launch Dashboard
```bash
cd crafter-ai-dashboard
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000) to view the live dashboard.

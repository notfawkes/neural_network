# Training Guide & Operational Runbook

Comprehensive guide for training, evaluating, resuming, and tuning CRAfter AI agents.

---

## 1. Hardware Execution & Device Selection

The engine automatically detects Apple Silicon MPS (Metal Performance Shaders), CUDA, or CPU:

```bash
python scripts/inspect_environment.py
```

### Apple Silicon M4 Benchmarks
- **MPS Throughput**: ~450 - 650 environment steps / sec across 4 parallel environments.
- **CPU Throughput**: ~220 - 300 steps / sec.
- In `configs/default.yaml`, `device.prefer_mps: true` is enabled by default.

---

## 2. Configuration Profiles

### CNN + LSTM Recurrent Agent (Default)
`configs/default.yaml`
```yaml
experiment:
  name: "crafter_lstm_ppo"
  model_type: "RecurrentActorCritic"
training:
  total_timesteps: 1000000
  num_envs: 4
  rollout_steps: 128
  ppo_epochs: 4
  learning_rate: 0.0003
  lstm_hidden_size: 256
```

### Feedforward Baseline
`configs/baseline.yaml`
```yaml
experiment:
  name: "crafter_mlp_ppo"
  model_type: "ActorCritic"
training:
  total_timesteps: 500000
  num_envs: 4
  rollout_steps: 128
  ppo_epochs: 4
  learning_rate: 0.0003
```

---

## 3. Training Commands

### Start Standard Run
```bash
python scripts/train.py --config configs/default.yaml
```

### Resume from Checkpoint
```bash
python scripts/train.py --config configs/default.yaml --resume checkpoints/crafter_lstm_ppo_latest.pt
```

### Run Custom Timesteps
```bash
python scripts/train.py --config configs/default.yaml --total-timesteps 250000
```

---

## 4. Evaluation & Checkpoint Testing

Evaluate a saved checkpoint over 50 test episodes across random world seeds:
```bash
python scripts/evaluate.py --checkpoint checkpoints/crafter_lstm_ppo_latest.pt --episodes 50
```

Interactive play session with terminal visualization:
```bash
python scripts/play.py --checkpoint checkpoints/crafter_lstm_ppo_latest.pt --episodes 3
```

Side-by-side policy comparison on identical observations:
```bash
python scripts/compare.py --checkpoint-a checkpoints/crafter_lstm_ppo_step00050000.pt --checkpoint-b checkpoints/crafter_lstm_ppo_step00250000.pt
```

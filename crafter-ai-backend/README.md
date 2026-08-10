# CRAfter AI Backend API

FastAPI backend server managing reinforcement learning training processes, streaming live observations over WebSockets, and serving episode replays and model evaluations.

---

## API Endpoints

### Training Management
- `POST /api/training/start`: Launch training subprocess.
- `POST /api/training/stop`: Gracefully stop running training.
- `POST /api/training/resume`: Resume training from a checkpoint.
- `GET /api/training/status`: Return running status and real-time step telemetry.
- `GET /api/training/metrics`: Retrieve historical returns, lengths, and PPO loss metrics.
- `GET /api/training/logs`: Retrieve recent stdout/stderr output lines.

### Episodes & Replay
- `GET /api/episodes`: List recorded episodes with pagination.
- `GET /api/episodes/{id}`: Detailed episode metadata and trajectory coordinates.
- `GET /api/episodes/{id}/frames`: Base64 JPEG frames for full observation playback.

### Checkpoints & Comparison
- `GET /api/models`: List all saved model checkpoints with file sizes and steps.
- `GET /api/models/{id}`: Checkpoint configuration and parameter overview.
- `POST /api/models/compare`: Evaluate two checkpoints on identical visual observations.

### Achievements & Experiments
- `GET /api/achievements`: Success rates across all 22 Crafter milestones.
- `GET /api/experiments`: List available YAML configurations.

### WebSockets
- `/ws/training`: Real-time training telemetry stream (500ms heartbeat).
- `/ws/live-game`: Observation frame stream with action distributions (10 FPS).

---

## Running the Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

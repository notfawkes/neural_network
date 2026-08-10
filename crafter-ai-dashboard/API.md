# Frontend API Integration Reference

This document maps all frontend data hooks and calls to backend REST endpoints.

---

| Method | Endpoint | Frontend Caller | Purpose |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/training/status` | `api.getTrainingStatus()` | Polls training status, current step, and speed |
| `GET` | `/api/training/metrics` | `api.getMetrics()` | Retrieves historical returns and update losses |
| `POST` | `/api/training/start` | `api.startTraining()` | Launches training subprocess |
| `POST` | `/api/training/stop` | `api.stopTraining()` | Terminates active training subprocess |
| `GET` | `/api/episodes` | `api.getEpisodes()` | Lists recorded gameplay episodes |
| `GET` | `/api/episodes/{id}` | `api.getEpisode(id)` | Fetches complete episode metadata and trajectory |
| `GET` | `/api/episodes/{id}/frames` | `api.getEpisodeFrames(id)` | Fetches base64-encoded frame sequence for replay |
| `GET` | `/api/models` | `api.getModels()` | Lists all stored checkpoints |
| `POST` | `/api/models/compare` | `api.compareModels()` | Evaluates two checkpoints on identical frames |
| `GET` | `/api/achievements` | `api.getAchievements()` | Calculates 22-achievement success rates |
| `GET` | `/api/experiments` | `api.getExperiments()` | Lists experiment YAML configurations |
| `WS` | `/ws/live-game` | `WebSocket` in `/live` | High-frequency frame and action distribution stream |

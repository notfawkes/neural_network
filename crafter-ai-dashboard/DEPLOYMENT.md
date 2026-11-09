# Deployment & Production Runbook

Guide for deploying CRAfter AI locally or on cloud workstations.

---

## Local Development (macOS / Apple Silicon)

1. **Start Backend**:
   ```bash
   cd crafter-ai-backend
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

2. **Start Dashboard**:
   ```bash
   cd crafter-ai-dashboard
   npm run dev
   ```

3. Open your browser to `http://localhost:3000`.

---

## Production Build

### Building the Dashboard
```bash
cd crafter-ai-dashboard
npm run build
npm run start
```

### Running Backend with Gunicorn/Uvicorn Workers
```bash
cd crafter-ai-backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | REST API base URL |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000/ws/live-game` | Live observation WebSocket URL |
| `CRAFTER_HOST` | `0.0.0.0` | Backend bind host |
| `CRAFTER_PORT` | `8000` | Backend bind port |


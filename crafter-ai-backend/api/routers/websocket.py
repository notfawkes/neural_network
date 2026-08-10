"""WebSocket handlers for streaming live training state and live agent POV."""
import io
import json
import base64
import asyncio
import logging
import numpy as np
from PIL import Image
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.config import settings
from services.metrics_service import MetricsService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])
metrics_service = MetricsService(settings.workspace_dir)

ACTION_NAMES = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
]


@router.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """Stream live training metrics every 500ms."""
    await websocket.accept()
    try:
        while True:
            state = metrics_service.get_live_state()
            await websocket.send_json(state)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        logger.info("Training websocket client disconnected")
    except Exception as e:
        logger.error(f"Error in training websocket: {e}")


@router.websocket("/ws/live-game")
async def websocket_live_game(websocket: WebSocket):
    """
    Stream live game POV observations and network decisions.
    Reads latest obs from RecordingStorage state directory.
    """
    await websocket.accept()
    live_obs_path = settings.workspace_dir / "experiments" / "state" / "crafter_lstm_ppo_live_obs.npy"

    try:
        while True:
            state = metrics_service.get_live_state()
            frame_b64 = None

            # Try to read live observation numpy file
            if live_obs_path.exists():
                try:
                    obs = np.load(str(live_obs_path))
                    # Check shape: if (3, 64, 64) float, convert to (64, 64, 3) uint8
                    if obs.shape[0] == 3 and obs.ndim == 3:
                        obs = np.transpose(obs, (1, 2, 0))
                        obs = np.clip(obs * 255.0, 0, 255).astype(np.uint8)
                    elif obs.dtype != np.uint8:
                        obs = np.clip(obs, 0, 255).astype(np.uint8)

                    img = Image.fromarray(obs).resize((320, 320), Image.NEAREST)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    frame_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                except Exception:
                    pass

            # If no live frame, generate placeholder or read latest episode frame
            if not frame_b64:
                # Look for most recent episode
                recent_episodes = sorted(settings.recordings_dir.glob("ep_*.npz"), reverse=True)
                if recent_episodes:
                    try:
                        data = np.load(str(recent_episodes[0]))
                        if "obs_frames" in data and len(data["obs_frames"]) > 0:
                            frame = data["obs_frames"][-1]
                            img = Image.fromarray(frame).resize((320, 320), Image.NEAREST)
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG", quality=85)
                            frame_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    except Exception:
                        pass

            # Mock or read current action distribution
            action_name = state.get("last_action", "noop")
            action_probs = state.get("action_probs")
            if not action_probs:
                # Uniform fallback with slight peak
                action_probs = [0.05] * 17
                action_probs[0] = 0.2

            payload = {
                "frame": frame_b64,
                "episode_id": state.get("episode_count", 0),
                "step": state.get("global_step", 0),
                "reward": state.get("last_reward", 0.0),
                "episode_reward": state.get("mean_reward", 0.0),
                "action": action_name,
                "action_probs": action_probs,
                "critic_value": state.get("value_loss", 0.0),
                "is_alive": state.get("status") == "training",
                "lstm_h_norm": state.get("lstm_h_norm", 1.0),
            }

            await websocket.send_json(payload)
            await asyncio.sleep(0.1)  # 10 FPS
    except WebSocketDisconnect:
        logger.info("Live-game websocket client disconnected")
    except Exception as e:
        logger.error(f"Error in live-game websocket: {e}")

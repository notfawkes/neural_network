"""Model service for managing checkpoints and side-by-side policy comparisons."""
import os
import io
import json
import base64
import logging
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

ACTION_NAMES = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
]


class ModelService:
    def __init__(self, checkpoints_dir: Path, recordings_dir: Path):
        self.checkpoints_dir = checkpoints_dir
        self.recordings_dir = recordings_dir
        self._device = torch.device("cpu")

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        checkpoints = []
        for ckpt_file in sorted(self.checkpoints_dir.glob("*.pt"), reverse=True):
            try:
                stat = ckpt_file.stat()
                # Parse metadata if possible without loading heavy weights every time
                name = ckpt_file.stem
                step = 0
                tag = "step"
                if "step" in name:
                    try:
                        step_part = name.split("step")[1].split("_")[0]
                        step = int(step_part)
                    except Exception:
                        pass
                elif "latest" in name:
                    tag = "latest"
                elif "best" in name:
                    tag = "best"
                elif "final" in name:
                    tag = "final"

                checkpoints.append({
                    "model_id": ckpt_file.name,
                    "filename": ckpt_file.name,
                    "path": str(ckpt_file),
                    "step": step,
                    "tag": tag,
                    "episode_count": 0,
                    "model_type": "RecurrentActorCritic" if "lstm" in name else "ActorCritic",
                    "size_bytes": stat.st_size,
                    "created_at": stat.st_mtime,
                })
            except Exception as e:
                logger.error(f"Error reading checkpoint {ckpt_file}: {e}")

        return checkpoints

    def get_checkpoint_detail(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint header and return configuration."""
        ckpt_path = self.checkpoints_dir / model_id
        if not ckpt_path.exists():
            return None

        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            return {
                "model_id": model_id,
                "global_step": ckpt.get("global_step", 0),
                "episode_count": ckpt.get("episode_count", 0),
                "update_count": ckpt.get("update_count", 0),
                "model_type": ckpt.get("model_type", "RecurrentActorCritic"),
                "config": ckpt.get("config", {}),
                "experiment_name": ckpt.get("experiment_name", "unknown"),
            }
        except Exception as e:
            logger.error(f"Error loading checkpoint {model_id}: {e}")
            return None

    def compare_checkpoints(
        self,
        checkpoint_a: str,
        checkpoint_b: str,
        obs_source_episode_id: Optional[int] = None,
        num_samples: int = 4,
    ) -> Dict[str, Any]:
        """Run policy comparison on both models with real observations."""
        from crafter_ai.models.actor_critic import ActorCritic, RecurrentActorCritic

        path_a = self._resolve_path(checkpoint_a)
        path_b = self._resolve_path(checkpoint_b)

        if not path_a.exists():
            raise FileNotFoundError(f"Checkpoint A not found: {checkpoint_a}")
        if not path_b.exists():
            raise FileNotFoundError(f"Checkpoint B not found: {checkpoint_b}")

        # Load weights
        ckpt_a = torch.load(str(path_a), map_location=self._device)
        ckpt_b = torch.load(str(path_b), map_location=self._device)

        model_a = self._instantiate_model(ckpt_a)
        model_b = self._instantiate_model(ckpt_b)

        model_a.load_state_dict(ckpt_a["model_state_dict"])
        model_b.load_state_dict(ckpt_b["model_state_dict"])
        model_a.eval()
        model_b.eval()

        # Get observations (from recorded episode or environment)
        raw_frames, obs_tensor = self._get_sample_observations(obs_source_episode_id, num_samples)

        # Run inference
        samples_result = []
        with torch.no_grad():
            for i in range(len(obs_tensor)):
                single_obs = obs_tensor[i : i + 1]  # (1, 3, 64, 64)

                # Inference on model A
                probs_a, val_a = self._run_inference(model_a, single_obs)
                # Inference on model B
                probs_b, val_b = self._run_inference(model_b, single_obs)

                top_a = ACTION_NAMES[int(np.argmax(probs_a))]
                top_b = ACTION_NAMES[int(np.argmax(probs_b))]

                entropy_a = float(-np.sum(probs_a * np.log(probs_a + 1e-8)))
                entropy_b = float(-np.sum(probs_b * np.log(probs_b + 1e-8)))

                # Encode frame to base64
                b64_frame = None
                if raw_frames is not None and i < len(raw_frames):
                    img = Image.fromarray(raw_frames[i]).resize((256, 256), Image.NEAREST)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    b64_frame = base64.b64encode(buf.getvalue()).decode("utf-8")

                action_comps = []
                for a_idx, name in enumerate(ACTION_NAMES):
                    pa = float(probs_a[a_idx])
                    pb = float(probs_b[a_idx])
                    action_comps.append({
                        "action_index": a_idx,
                        "action_name": name,
                        "prob_a": pa,
                        "prob_b": pb,
                        "delta": pb - pa,
                    })

                samples_result.append({
                    "sample_index": i,
                    "frame_base64": b64_frame,
                    "value_a": val_a,
                    "value_b": val_b,
                    "entropy_a": entropy_a,
                    "entropy_b": entropy_b,
                    "top_action_a": top_a,
                    "top_action_b": top_b,
                    "actions": action_comps,
                })

        return {
            "model_a_info": {
                "id": path_a.name,
                "step": ckpt_a.get("global_step", 0),
                "model_type": ckpt_a.get("model_type", "RecurrentActorCritic"),
            },
            "model_b_info": {
                "id": path_b.name,
                "step": ckpt_b.get("global_step", 0),
                "model_type": ckpt_b.get("model_type", "RecurrentActorCritic"),
            },
            "samples": samples_result,
        }

    def _resolve_path(self, identifier: str) -> Path:
        p = Path(identifier)
        if p.is_file():
            return p
        cand = self.checkpoints_dir / identifier
        if cand.is_file():
            return cand
        cand_pt = self.checkpoints_dir / f"{identifier}.pt"
        if cand_pt.is_file():
            return cand_pt
        return cand

    def _instantiate_model(self, ckpt: Dict[str, Any]):
        from crafter_ai.models.actor_critic import ActorCritic, RecurrentActorCritic
        model_type = ckpt.get("model_type", "RecurrentActorCritic")
        config = ckpt.get("config", {})
        if model_type == "RecurrentActorCritic":
            return RecurrentActorCritic(
                in_channels=3,
                num_actions=17,
                cnn_out=config.get("cnn_out_features", 512),
                lstm_hidden=config.get("lstm_hidden_size", 256),
            ).to(self._device)
        else:
            return ActorCritic(
                in_channels=3,
                num_actions=17,
                cnn_out=config.get("cnn_out_features", 512),
            ).to(self._device)

    def _run_inference(self, model, obs_tensor: torch.Tensor):
        from crafter_ai.models.actor_critic import RecurrentActorCritic
        if isinstance(model, RecurrentActorCritic):
            state = model.initial_state(1, self._device)
            done = torch.zeros(1, device=self._device)
            _, _, _, value, probs, _ = model.get_action_and_value(obs_tensor, done, state)
            return probs[0].cpu().numpy(), float(value[0].item())
        else:
            _, _, _, value, probs = model.get_action_and_value(obs_tensor)
            return probs[0].cpu().numpy(), float(value[0].item())

    def _get_sample_observations(self, episode_id: Optional[int], num_samples: int):
        from crafter_ai.env.adapter import CrafterEnv
        raw_frames = None
        if episode_id is not None:
            ep_file = self.recordings_dir / f"ep_{episode_id:08d}.npz"
            if ep_file.exists():
                data = np.load(str(ep_file))
                if "obs_frames" in data and len(data["obs_frames"]) > 0:
                    frames = data["obs_frames"]
                    indices = np.linspace(0, len(frames) - 1, min(num_samples, len(frames)), dtype=int)
                    raw_frames = frames[indices]
                    obs = CrafterEnv.preprocess_batch(raw_frames)
                    return raw_frames, torch.as_tensor(obs, dtype=torch.float32)

        # Fallback to fresh Crafter environment
        env = CrafterEnv(reward=True, seed=42)
        raw_list = []
        obs_list = []
        obs, _ = env.reset()
        raw_list.append(env.last_raw_obs)
        obs_list.append(obs)
        for _ in range(num_samples - 1):
            for _ in range(30):
                obs, _, term, trunc, _ = env.step(np.random.randint(0, 17))
                if term or trunc:
                    obs, _ = env.reset()
            raw_list.append(env.last_raw_obs)
            obs_list.append(obs)

        return np.stack(raw_list), torch.as_tensor(np.stack(obs_list), dtype=torch.float32)


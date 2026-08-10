"""Training service coordinating process management and training state."""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from core.process_manager import ProcessManager
from .metrics_service import MetricsService

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(self, process_manager: ProcessManager, metrics_service: MetricsService):
        self.pm = process_manager
        self.ms = metrics_service

    def start(
        self,
        config_path: str = "configs/default.yaml",
        resume_checkpoint: Optional[str] = None,
        total_timesteps: Optional[int] = None,
        experiment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.pm.start_training(
            config_path=config_path,
            resume_checkpoint=resume_checkpoint,
            total_timesteps=total_timesteps,
            experiment_name=experiment_name,
        )

    def stop(self) -> Dict[str, Any]:
        return self.pm.stop_training()

    def get_status(self) -> Dict[str, Any]:
        live_state = self.ms.get_live_state()
        is_running = self.pm.is_running
        pid = self.pm.pid

        # Override or harmonize status
        if is_running:
            live_state["status"] = "training"
        elif live_state.get("status") == "training":
            live_state["status"] = "stopped"

        live_state["is_running"] = is_running
        live_state["pid"] = pid
        return live_state

    def get_metrics(self) -> Dict[str, Any]:
        return self.ms.get_full_metrics()

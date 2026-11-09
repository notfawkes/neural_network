"""
Process Manager
===============
Manages background training subprocesses safely and reliably.
Provides start, stop, monitor, and log retrieval functions.
"""

import os
import sys
import time
import signal
import psutil
import subprocess
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class ProcessManager:
    """Manages training subprocess execution and lifecycle."""

    def __init__(self, workspace_dir: Path, logs_dir: Path):
        self.workspace_dir = workspace_dir
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._current_experiment: str = "default"
        self._log_file_path: Optional[Path] = None
        self._start_time: float = 0.0

    @property
    def is_running(self) -> bool:
        with self._lock:
            if self._process is None:
                return False
            poll = self._process.poll()
            return poll is None

    @property
    def pid(self) -> Optional[int]:
        with self._lock:
            if self._process and self._process.poll() is None:
                return self._process.pid
            return None

    def start_training(
        self,
        config_path: str = "configs/default.yaml",
        resume_checkpoint: Optional[str] = None,
        total_timesteps: Optional[int] = None,
        experiment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Launch the training script in a dedicated subprocess."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(f"Training is already running with PID {self._process.pid}")

            exp_name = experiment_name or "crafter_lstm_ppo"
            self._current_experiment = exp_name
            self._start_time = time.time()

            log_filename = f"train_{exp_name}_{int(self._start_time)}.log"
            self._log_file_path = self.logs_dir / log_filename
            log_file = open(self._log_file_path, "w", buffering=1)

            # Determine python executable & engine script
            python_exe = sys.executable
            engine_dir = self.workspace_dir / "crafter-ai-engine"
            train_script = engine_dir / "scripts" / "train.py"

            resolved_config = config_path
            if not os.path.isabs(resolved_config):
                resolved_config = str(engine_dir / config_path)

            cmd = [python_exe, str(train_script), "--config", str(resolved_config)]
            if resume_checkpoint:
                cmd.extend(["--resume", str(resume_checkpoint)])
            if total_timesteps:
                cmd.extend(["--total-timesteps", str(total_timesteps)])

            env = os.environ.copy()
            # Ensure engine & crafter root are on PYTHONPATH
            python_path = f"{engine_dir}:{self.workspace_dir}:{env.get('PYTHONPATH', '')}"
            env["PYTHONPATH"] = python_path

            logger.info(f"Starting training process: {' '.join(cmd)}")
            self._process = subprocess.Popen(
                cmd,
                cwd=str(engine_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )

            return {
                "status": "started",
                "pid": self._process.pid,
                "command": cmd,
                "experiment_name": exp_name,
                "log_file": str(self._log_file_path),
                "start_time": self._start_time,
            }

    def stop_training(self, timeout_sec: float = 5.0) -> Dict[str, Any]:
        """Gracefully stop (SIGINT/SIGTERM) then force kill if needed."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return {"status": "not_running", "pid": None}

            pid = self._process.pid
            logger.info(f"Stopping training process PID {pid}...")

            try:
                # Send SIGINT first to trigger graceful checkpointing
                if hasattr(os, "killpg"):
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGINT)
                else:
                    self._process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass

            # Wait for exit
            t0 = time.time()
            while time.time() - t0 < timeout_sec:
                if self._process.poll() is not None:
                    logger.info(f"Process {pid} exited gracefully.")
                    return {"status": "stopped", "pid": pid, "graceful": True}
                time.sleep(0.2)

            # Force terminate
            logger.warning(f"Process {pid} did not terminate within {timeout_sec}s. Terminating...")
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                else:
                    self._process.terminate()
            except ProcessLookupError:
                pass

            time.sleep(1.0)
            if self._process.poll() is None:
                logger.error(f"Force killing process {pid} (SIGKILL)...")
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    else:
                        self._process.kill()
                except ProcessLookupError:
                    pass

            return {"status": "killed", "pid": pid, "graceful": False}

    def get_recent_logs(self, max_lines: int = 100) -> List[str]:
        """Return the latest N lines of stdout/stderr logs."""
        if not self._log_file_path or not self._log_file_path.exists():
            return []
        try:
            with open(self._log_file_path, "r", errors="replace") as f:
                lines = f.readlines()
                return [line.rstrip() for line in lines[-max_lines:]]
        except Exception as e:
            logger.error(f"Error reading logs: {e}")
            return []


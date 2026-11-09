"""Unit tests for FigureGenerator."""
import shutil
import tempfile
from pathlib import Path
import pytest
from crafter_ai.evaluation.figures import FigureGenerator


@pytest.fixture
def temp_figure_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_figure_generator_initialization(temp_figure_dir):
    gen = FigureGenerator(output_dir=str(temp_figure_dir))
    assert gen.output_dir.exists()


def test_figure_generator_learning_curves(temp_figure_dir):
    gen = FigureGenerator(output_dir=str(temp_figure_dir))
    metrics = {
        "episode_rewards": [0.5, 1.0, 1.5, 2.0, 2.5],
        "episode_lengths": [50, 80, 120, 150, 200],
        "episode_steps": [50, 130, 250, 400, 600],
    }
    p = gen.plot_learning_curves(metrics)
    assert p.exists()
    assert p.stat().st_size > 1000


def test_figure_generator_losses(temp_figure_dir):
    gen = FigureGenerator(output_dir=str(temp_figure_dir))
    metrics = {
        "update_metrics": [
            {"policy_loss": -0.01, "value_loss": 0.4, "entropy_loss": 2.5, "approx_kl": 0.005, "global_step": 512},
            {"policy_loss": -0.02, "value_loss": 0.3, "entropy_loss": 2.3, "approx_kl": 0.004, "global_step": 1024},
        ]
    }
    p = gen.plot_ppo_losses(metrics)
    assert p.exists()
    assert p.stat().st_size > 1000


def test_figure_generator_generate_all(temp_figure_dir):
    gen = FigureGenerator(output_dir=str(temp_figure_dir))
    figs = gen.generate_all_figures(experiment_name="test_exp")
    assert len(figs) == 10
    for f in figs:
        assert f.exists()
        assert f.stat().st_size > 1000


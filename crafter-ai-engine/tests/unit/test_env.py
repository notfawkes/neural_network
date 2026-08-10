"""
Unit tests for the Crafter environment adapter.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from crafter_ai.env.adapter import (
    CrafterEnv, VecCrafterEnv, make_env,
    ACTION_NAMES, ACHIEVEMENT_NAMES, NUM_ACTIONS, NUM_ACHIEVEMENTS,
    OBS_SHAPE_CHW,
)


class TestCrafterEnv:
    """Tests for the single CrafterEnv adapter."""

    def setup_method(self):
        self.env = CrafterEnv(reward=True, seed=42)

    def test_reset_returns_correct_shape(self):
        obs, info = self.env.reset()
        assert obs.shape == OBS_SHAPE_CHW, f"Expected {OBS_SHAPE_CHW}, got {obs.shape}"

    def test_reset_returns_float32(self):
        obs, info = self.env.reset()
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    def test_reset_obs_in_range(self):
        obs, info = self.env.reset()
        assert obs.min() >= 0.0, "Obs should be >= 0"
        assert obs.max() <= 1.0, "Obs should be <= 1"

    def test_reset_info_has_required_keys(self):
        _, info = self.env.reset()
        required_keys = ["episode_id", "episode_step", "episode_reward", "achievements"]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_step_returns_correct_shapes(self):
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(0)
        assert obs.shape == OBS_SHAPE_CHW
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_obs_in_range(self):
        self.env.reset()
        obs, _, _, _, _ = self.env.step(1)
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_all_actions_valid(self):
        """All 17 actions should be accepted without error."""
        self.env.reset()
        for action in range(NUM_ACTIONS):
            self.env.reset()  # Reset to avoid death
            obs, reward, terminated, truncated, info = self.env.step(action)
            assert obs.shape == OBS_SHAPE_CHW

    def test_action_space_size(self):
        assert self.env.num_actions == 17

    def test_action_names_correct(self):
        assert len(self.env.action_names) == 17
        assert "noop" in self.env.action_names
        assert "move_left" in self.env.action_names
        assert "do" in self.env.action_names

    def test_achievement_names(self):
        assert len(ACHIEVEMENT_NAMES) == 22
        assert "collect_wood" in ACHIEVEMENT_NAMES
        assert "make_iron_sword" in ACHIEVEMENT_NAMES
        assert "wake_up" in ACHIEVEMENT_NAMES

    def test_episode_step_increments(self):
        self.env.reset()
        for i in range(5):
            _, _, _, _, info = self.env.step(0)
            assert info["episode_step"] == i + 1

    def test_episode_id_increments_on_reset(self):
        _, info1 = self.env.reset()
        ep1 = info1["episode_id"]
        _, info2 = self.env.reset()
        ep2 = info2["episode_id"]
        assert ep2 > ep1

    def test_raw_obs_stored_when_enabled(self):
        env = CrafterEnv(reward=True, record_raw=True)
        env.reset()
        assert env.last_raw_obs is not None
        assert env.last_raw_obs.shape == (64, 64, 3)
        assert env.last_raw_obs.dtype == np.uint8

    def test_player_position_returned(self):
        self.env.reset()
        _, _, _, _, info = self.env.step(0)
        pos = info.get("player_pos")
        assert pos is not None
        assert len(pos) == 2

    def test_trajectory_tracking(self):
        self.env.reset()
        for _ in range(10):
            self.env.step(1)  # move_left
        traj = self.env.get_trajectory()
        assert len(traj) >= 10

    def test_preprocessing_correct(self):
        """Test that _preprocess converts HWC uint8 to CHW float32."""
        raw = np.zeros((64, 64, 3), dtype=np.uint8)
        raw[0, 0, 0] = 255  # Set one pixel
        processed = CrafterEnv._preprocess(raw)
        assert processed.shape == (3, 64, 64)
        assert processed.dtype == np.float32
        assert processed[0, 0, 0] == pytest.approx(1.0)
        assert processed[1, 0, 0] == pytest.approx(0.0)


class TestVecCrafterEnv:
    """Tests for the vectorized environment."""

    def setup_method(self):
        self.num_envs = 4
        self.env = VecCrafterEnv(num_envs=self.num_envs, reward=True)

    def test_reset_batch_shape(self):
        obs, infos = self.env.reset()
        assert obs.shape == (self.num_envs, 3, 64, 64)
        assert len(infos) == self.num_envs

    def test_step_batch_shape(self):
        self.env.reset()
        actions = np.zeros(self.num_envs, dtype=np.int64)
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        assert obs.shape == (self.num_envs, 3, 64, 64)
        assert rewards.shape == (self.num_envs,)
        assert terminated.shape == (self.num_envs,)
        assert truncated.shape == (self.num_envs,)
        assert len(infos) == self.num_envs

    def test_step_different_actions(self):
        self.env.reset()
        actions = np.array([0, 1, 2, 3])
        obs, rewards, _, _, _ = self.env.step(actions)
        assert obs.shape == (self.num_envs, 3, 64, 64)

    def test_make_env_factory(self):
        env = make_env(num_envs=2, reward=True, seed=0)
        obs, _ = env.reset()
        assert obs.shape == (2, 3, 64, 64)


class TestPreprocessBatch:
    def test_preprocess_batch(self):
        batch = np.ones((4, 64, 64, 3), dtype=np.uint8) * 128
        processed = CrafterEnv.preprocess_batch(batch)
        assert processed.shape == (4, 3, 64, 64)
        assert processed.dtype == np.float32
        expected = 128 / 255.0
        np.testing.assert_allclose(processed, expected, atol=1e-6)


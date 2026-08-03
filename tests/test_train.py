"""Trainer plumbing: returns, eval metrics, and what a checkpoint records.

Not a test of whether the snake learns -- that is stochastic and slow. These
cover the parts that quietly produce misleading numbers when wrong.
"""

import matplotlib

matplotlib.use("Agg")

import pytest
import torch

from snake_rl.env import SnakeEnv
from snake_rl.models import SnakePolicy
from snake_rl.train import compute_returns, evaluate, train


def tiny_policy(env):
  return SnakePolicy(observation_dim=env.observation_dim, hidden_dim=8)


class TestComputeReturns:
  def test_undiscounted_returns_are_suffix_sums(self):
    assert compute_returns([1.0, 2.0, 3.0], gamma=1.0) == pytest.approx(
      [6.0, 5.0, 3.0])

  def test_discounting_shrinks_later_rewards(self):
    returns = compute_returns([0.0, 0.0, 10.0], gamma=0.5)
    assert returns == pytest.approx([2.5, 5.0, 10.0])

  def test_empty_episode_gives_empty_returns(self):
    assert len(compute_returns([], gamma=0.99)) == 0


class TestEvaluate:
  def test_reports_the_expected_metrics(self):
    env = SnakeEnv(seed=0)
    metrics = evaluate(tiny_policy(env), seed=0, episodes=3)
    assert set(metrics) == {"return", "length", "score", "foods",
                            "steps_per_food", "death_rate"}
    assert 0.0 <= metrics["death_rate"] <= 1.0
    assert metrics["length"] >= 3.0



  def test_evaluates_on_the_env_it_is_given(self):
    """Eval scores the env it is handed, not a default one built on the spot."""
    env_kwargs = {"horizon": 12}
    env = SnakeEnv(seed=0, **env_kwargs)
    metrics = evaluate(tiny_policy(env), seed=0, episodes=4,
                       env_kwargs=env_kwargs)
    # With a 12-step horizon no episode can run long enough to average more
    # than 12 steps per food, let alone the ~300 a default env allows.
    assert metrics["length"] < 8

  def test_steps_per_food_is_infinite_when_nothing_is_eaten(self):
    env_kwargs = {"horizon": 2}
    env = SnakeEnv(seed=0, **env_kwargs)
    metrics = evaluate(tiny_policy(env), seed=0, episodes=3,
                       env_kwargs=env_kwargs)
    assert metrics["foods"] == 0
    assert metrics["steps_per_food"] == float("inf")


class TestTrainSmoke:
  def test_runs_and_records_the_env_config(self, tmp_path):
    checkpoint = tmp_path / "policy.pt"
    train(episodes=3, seed=0, gamma=0.99, entropy_bonus=0.03,
          checkpoint=str(checkpoint), plot="", hidden_dim=8,
          step_cost=0.02, horizon=120,
          quiet=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # The viewer rebuilds its env from this, so it has to be complete.
    assert saved["env_kwargs"]["step_cost"] == 0.02
    assert saved["env_kwargs"]["horizon"] == 120
    assert saved["observation_dim"] == 18

  def test_saves_a_training_curve(self, tmp_path):
    plot = tmp_path / "curve.png"
    train(episodes=2, seed=0, gamma=0.99, entropy_bonus=0.03,
          checkpoint="", plot=str(plot), hidden_dim=8, quiet=True)
    assert plot.exists() and plot.stat().st_size > 0


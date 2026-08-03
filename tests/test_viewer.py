"""Viewer wiring: the parts that silently produce a wrong GIF when broken.

The rendering itself is not worth asserting on, but the viewer has to agree
with the env about the rules of the game and about how an episode ended, and
both of those have been wrong before.
"""

import matplotlib

matplotlib.use("Agg")  # no display in CI, and these tests never show a window

import pytest
import torch

from snake_rl.env import TURN_LEFT, SnakeEnv
from snake_rl.models import SnakePolicy
from snake_rl.viewer import SnakeViewer


def make_checkpoint(path, env_kwargs, hidden_dim=16):
  """Write a checkpoint shaped the way `snake_rl/train.py` writes one."""
  env = SnakeEnv(seed=0, **env_kwargs)
  policy = SnakePolicy(observation_dim=env.observation_dim,
                       hidden_dim=hidden_dim)
  torch.save({"policy_state_dict": policy.state_dict(), "seed": 0,
              "episodes": 1, "hidden_dim": hidden_dim,
              "observation_dim": env.observation_dim,
              "env_kwargs": env_kwargs}, path)
  return path


class TestOutcome:
  """The ending label is read off `info`, not off the reward.

  A `reward == -10.0` test cannot tell the endings apart: starving pays
  exactly what crashing pays, so every starvation would be labelled a crash.
  """

  def test_crash_is_labelled_crashed(self):
    viewer = SnakeViewer(random_actions=True, stop_at_death=True, seed=0)
    viewer.env.snake = [(0, 3), (0, 2), (0, 1)]
    viewer.env.direction = 1  # RIGHT
    viewer.env.food = (5, 5)
    viewer.observation = viewer.env._observation()
    for _ in range(200):
      viewer.update(0)
      if viewer.finished:
        break
    assert viewer.outcome == "CRASHED"

  def test_hitting_the_horizon_is_labelled_out_of_time(self):
    """Circling on a big board: alive at the horizon, so not a crash."""
    viewer = SnakeViewer(stop_at_death=True, seed=0, horizon=8)
    # Drive it by hand so the ending is the horizon and nothing else.
    viewer._decide = lambda: (TURN_LEFT, viewer.last_probabilities)
    for _ in range(30):
      viewer.update(0)
      if viewer.finished:
        break
    assert viewer.outcome == "OUT OF TIME"

  def test_starving_is_labelled_starved(self):
    """Circling past the starvation clock, still alive: its own ending."""
    viewer = SnakeViewer(stop_at_death=True, seed=0, horizon=10_000)
    viewer.env.food = (0, 0)  # a corner it will never reach while circling
    viewer._decide = lambda: (TURN_LEFT, viewer.last_probabilities)
    for _ in range(500):
      viewer.update(0)
      if viewer.finished:
        break
    assert viewer.outcome == "STARVED"


class TestCheckpointConfig:
  def test_env_is_rebuilt_from_the_checkpoint_config(self, tmp_path):
    """A policy must be replayed on the game it was trained on, not on
    whatever the current defaults are."""
    path = make_checkpoint(tmp_path / "cfg.pt",
                           {"grid_size": 10, "step_cost": 0.05})
    viewer = SnakeViewer(checkpoint=str(path), seed=0)
    assert viewer.env.grid_size == 10
    assert viewer.env.step_cost == 0.05
    # The real check: it can actually take a step without a shape error.
    viewer.update(0)

  def test_horizon_flag_still_wins_over_the_checkpoint(self, tmp_path):
    path = make_checkpoint(tmp_path / "h.pt", {"horizon": 300})
    viewer = SnakeViewer(checkpoint=str(path), seed=0, horizon=42)
    assert viewer.env.horizon == 42

  def test_committed_checkpoint_still_replays(self):
    """Whatever checkpoint the repo ships must load and step cleanly.

    Deliberately does not pin an observation size: the committed policy gets
    retrained as better configurations are found, and this should not need
    editing when it does.
    """
    import os
    if not os.path.exists("trained_policy.pt"):
      pytest.skip("no committed checkpoint in this working tree")
    viewer = SnakeViewer(checkpoint="trained_policy.pt", seed=0)
    # The env built from the checkpoint has to match the net it feeds.
    assert viewer.env.observation_dim == viewer.policy.net[0].in_features
    viewer.update(0)



class TestStatusLine:
  def test_status_reports_length_score_and_step(self):
    viewer = SnakeViewer(random_actions=True, seed=0)
    viewer.update(0)
    text = viewer.status.get_text()
    assert "length" in text and "score" in text and "step" in text

  def test_score_is_shown_after_eating(self):
    viewer = SnakeViewer(random_actions=True, seed=0)
    head_row, head_col = viewer.env.snake[0]
    viewer.env.food = (head_row, head_col + 1)
    viewer.env.direction = 1  # RIGHT, straight at it
    viewer.observation = viewer.env._observation()
    viewer.update(0)
    assert viewer.env.score > 0
    assert f"score {viewer.env.score}" in viewer.status.get_text()


def test_gif_render_end_to_end(tmp_path):
  """The smoke test from the README, as an actual test."""
  from matplotlib.animation import FuncAnimation
  viewer = SnakeViewer(random_actions=True, seed=1, stop_at_death=True,
                       horizon=30)
  out = tmp_path / "smoke.gif"
  animation = FuncAnimation(
    viewer.figure, viewer.update, frames=viewer.episode_frames(40),
    save_count=40, blit=False, cache_frame_data=False,
    init_func=lambda: [viewer.image_artist, *viewer.bars, viewer.status])
  animation.save(out, writer="pillow", fps=8)
  assert out.exists() and out.stat().st_size > 0

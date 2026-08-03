"""The guard rails around the challenge, not the challenge itself.

`tests/test_env.py` and friends pin the *rules of the game*. These pin the
handful of things the graded suite has to be able to assume in order to grade
anything at all: that the observation positions the hand-written players read
still hold what the README says they hold, that the viewer draws a board that
can be read back, and that `train()` can be called and its policy scored.

Every one of these passes on the repository as shipped, bugs and all, and has
to keep passing on any solution. When one of them fails, the message to take
from it is "this fork moved something the challenge is built on", not "this
issue is unsolved" -- which is exactly why they live outside the `challenge`
marker and are never part of the scorecard.
"""

import matplotlib

matplotlib.use("Agg")  # no display in CI, and these tests never show a window

import numpy as np
import pytest

from snake_rl.env import STRAIGHT, TURN_LEFT, TURN_RIGHT, SnakeEnv

from .harness import (
  DANGER_BIT,
  FOOD_FORWARD,
  FOOD_RIGHT,
  board_image,
  draw,
  head_after,
  move_is_fatal,
  score_policy,
  staged_snake,
  step,
  train_policy,
)

ACTION_NAMES = {TURN_LEFT: "turn left", STRAIGHT: "go straight",
                TURN_RIGHT: "turn right"}


def walk(states=250, seed=1):
  """Random play, restarting on death: states worth measuring against."""
  rng = np.random.default_rng(seed)
  env = SnakeEnv(seed=seed)
  observation = env.reset()
  for _ in range(states):
    yield env, np.asarray(observation, dtype=float)
    observation, _, done, _ = step(env, int(rng.integers(env.action_dim)))
    if done:
      observation = env.reset()


def test_the_observation_is_wide_enough_for_the_documented_layout():
  # No issue asks for a narrower observation, and the graded suite reads
  # positions up to and including the free-space block.
  assert SnakeEnv.action_dim == 3
  assert SnakeEnv.observation_dim >= 18


@pytest.mark.parametrize("action", sorted(DANGER_BIT))
def test_each_danger_bit_warns_about_its_own_move(action):
  """The three danger bits still sit where the players look for them.

  `harness.DANGER_BIT` maps an action to the observation position that warns
  about it, and both hand-written players steer by that map. If the bits move
  or get reordered, the players stop working for a reason that has nothing to
  do with any open issue.
  """
  position = DANGER_BIT[action]
  disagreements = []
  for env, observation in walk():
    warned = observation[position] > 0.5
    fatal = move_is_fatal(env, action)
    if warned != fatal:
      disagreements.append((len(env.snake), warned, fatal))
  assert not disagreements, (
      f"observation[{position}] is supposed to warn about '"
      f"{ACTION_NAMES[action]}', and disagreed with what actually happens on "
      f"{len(disagreements)} states (first few: {disagreements[:3]}). The "
      "danger bits have moved or been reordered.")


def test_the_food_projection_still_sits_where_the_players_read_it():
  """Positions 11 and 12 hold the food in the snake's own frame.

  Recomputed from public state only -- the forward vector is read off where
  the head actually lands when the snake goes straight, and the right-hand
  vector is that turned ninety degrees -- so this checks the meaning rather
  than re-running the environment's own arithmetic.
  """
  checked, wrong = 0, []
  for env, observation in walk():
    ahead = head_after(env, STRAIGHT)
    if ahead is None:
      continue  # straight is fatal here; no forward vector to read off
    head_row, head_col = env.snake[0]
    forward = (ahead[0] - head_row, ahead[1] - head_col)
    right = (forward[1], -forward[0])  # ninety degrees clockwise
    offset = (env.food[0] - head_row, env.food[1] - head_col)

    expected_forward = (offset[0] * forward[0] + offset[1] * forward[1]) / env.grid_size
    expected_right = (offset[0] * right[0] + offset[1] * right[1]) / env.grid_size
    checked += 1
    if not (np.isclose(observation[FOOD_FORWARD], expected_forward, atol=1e-5)
            and np.isclose(observation[FOOD_RIGHT], expected_right, atol=1e-5)):
      wrong.append((observation[FOOD_FORWARD], expected_forward,
                    observation[FOOD_RIGHT], expected_right))

  assert checked > 50, "not enough survivable states to measure on"
  assert not wrong, (
      f"observation[{FOOD_FORWARD}] and observation[{FOOD_RIGHT}] are supposed "
      "to hold the food projected onto the heading and its right-hand normal, "
      f"and disagreed on {len(wrong)} of {checked} states (first: got "
      f"{wrong[0][0]:.4f}/{wrong[0][2]:.4f}, expected "
      f"{wrong[0][1]:.4f}/{wrong[0][3]:.4f}).")


def test_three_consecutive_channels_report_free_space():
  """Something in the observation varies with the room a move leads into.

  Deliberately weak: *which* three and in *what order* is issue #2's business.
  This only asks that the sensors exist and are not constant, because a
  constant sensor makes the graded measurement meaningless rather than failed.
  """
  observations = np.array([observation for _, observation in walk()])
  spreads = observations.std(axis=0)
  bounded = ((observations >= -0.01) & (observations <= 1.01)).all(axis=0)
  candidates = [index for index in range(len(spreads) - 2)
                if bounded[index:index + 3].all()
                and (spreads[index:index + 3] > 0.01).all()]
  assert candidates, (
      "no three consecutive observation channels stay within [0, 1] and vary "
      "from state to state, so there is nothing that looks like a free-space "
      "sensor to grade.\n  per-channel spread: "
      + " ".join(f"{value:.3f}" for value in spreads))


def test_the_viewer_draws_a_board_that_can_be_read_back():
  """The graded suite reads issue #1 off the rendered image, so it has to exist."""
  from snake_rl.viewer import SnakeViewer

  viewer = SnakeViewer(seed=3, stop_at_death=True, horizon=800)
  env = staged_snake(viewer, length=14)
  draw(viewer)
  image = board_image(viewer)

  assert image.shape[:2] == (env.grid_size, env.grid_size), (
      f"the board image is {image.shape[:2]}, not the "
      f"{(env.grid_size, env.grid_size)} grid it is drawing")
  background = image[0, 0]
  drawn = sum(1 for row, col in viewer.env.snake
              if not np.allclose(image[row, col], background, atol=1e-6))
  assert drawn >= len(viewer.env.snake) - 1, (
      f"only {drawn} of {len(viewer.env.snake)} snake cells are painted at "
      "all; the board is not being drawn, so nothing about the head marker "
      "can be measured")


def test_training_can_be_called_and_its_policy_scored():
  """Two episodes, purely to prove the graded suite can drive `train()`.

  Says nothing about whether training *works* -- that is issue #3, and it has
  to stay failing here on the repository as shipped.
  """
  import torch

  torch.set_num_threads(2)
  policy = train_policy(episodes=2, seed=0)
  metrics = score_policy(policy, 0, episodes=2)
  for key in ("return", "length", "death_rate"):
    assert key in metrics, f"evaluate() no longer reports {key!r}"
    assert np.isfinite(metrics[key]) or key == "steps_per_food"

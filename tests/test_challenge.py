"""Graded tests for the py-eats challenge issues.

One section per issue. Each test asserts exactly the issue's stated "Done
when" and nothing more -- a test that demands more than the issue asked for
fails honest work that solved it a different way.

Every test in here is meant to **fail on the repository as shipped**. That is
the point: they are the to-do list. `pytest -m challenge` runs just these;
`pytest -m "not challenge"` runs the guard rails on their own.

Read the failure messages. They are written to say what was measured and what
the bar was, because "assert False" tells a first-year nothing.
"""

import numpy as np
import pytest

from snake_rl.env import SnakeEnv

from .harness import (
  ACTIONS,
  FREE_SPACE_COUNT,
  board_image,
  body_colour_steps,
  describe_layout,
  draw,
  free_space_layout,
  rules_table,
  sample_states,
  score_policy,
  score_rules,
  staged_snake,
  train_policy,
  untrained_policy,
)

pytestmark = pytest.mark.challenge


# --- issue #1: the dark square is on the wrong end of the snake --------------
#
# The board is drawn as a colour gradient down the body with one cell
# overpainted to mark the head. Reading that back needs to know neither the
# marker's colour nor the gradient's: measure how far the colour moves between
# neighbouring body cells and exactly one of those steps will be large. Which
# end it sits at is the entire issue.
#
# A real marker measures between 6x and 17x the median step at these lengths,
# so anything above ~2.5 is an overpainted cell rather than gradient noise.
# Both lengths are deliberately long: on a five-cell snake the gradient itself
# steps by as much as a marker does, and the picture stops being able to answer
# the question at all.

MARKER_DOMINANCE = 2.5
SNAKE_LENGTHS = (14, 24)


@pytest.fixture(scope="module", params=SNAKE_LENGTHS, ids=lambda n: f"length{n}")
def drawn_board(request):
  """A staged board of a known snake length, rendered once."""
  from snake_rl.viewer import SnakeViewer

  viewer = SnakeViewer(seed=3, stop_at_death=True, horizon=800)
  staged_snake(viewer, length=request.param)
  draw(viewer)
  # `snake` is read back after drawing rather than before: `draw()` falls back
  # to `update()` on a viewer that renamed `_draw`, and that steps the game.
  return board_image(viewer), list(viewer.env.snake)


@pytest.mark.issue(1)
def test_one_snake_cell_is_painted_as_a_head_marker(drawn_board):
  """There is a marker at all -- the measurement the next test rests on.

  A body drawn as a plain gradient with nothing overpainted would leave the
  next test picking the largest of several equal steps, which is a coin flip.
  Fails here instead, saying so.
  """
  image, snake = drawn_board
  steps = body_colour_steps(image, snake)
  biggest = float(np.max(steps))
  median = float(np.median(steps))
  ratio = biggest / max(median, 1e-9)
  assert ratio > MARKER_DOMINANCE, (
      f"no snake cell stands out from the body gradient (largest colour step "
      f"{biggest:.3f}, median {median:.3f}, ratio {ratio:.1f}x -- "
      f"{MARKER_DOMINANCE}x is the bar). The board is supposed to overpaint "
      "one cell to mark the head; nothing here is overpainting anything.")


@pytest.mark.issue(1)
def test_the_head_marker_is_on_the_head_not_the_tail(drawn_board):
  """Issue #1's "Done when", as a measurement: the marker leads, not trails.

  Says nothing about *which* colour the marker is or which way the gradient
  runs -- only that the overpainted cell is `snake[0]`, the end moving into
  open space.
  """
  image, snake = drawn_board
  steps = body_colour_steps(image, snake)
  marker_edge = int(np.argmax(steps))
  where = ("the head end" if marker_edge == 0 else
           "the tail end" if marker_edge == len(steps) - 1 else
           f"{marker_edge} cells back from the head")
  assert marker_edge == 0, (
      f"the marked cell is at {where} of a {len(snake)}-cell snake. It has to "
      "sit at the front -- the end moving into open space -- so the picture "
      "reads as a snake going somewhere.\n"
      f"  colour steps head->tail: "
      + " ".join(f"{value:.3f}" for value in steps))


# --- issue #2: one of the two hand-written players can no longer survive -----
#
# The environment hands the policy three "free space" readings, one per move,
# and they are meant to arrive in the same turn left / straight / turn right
# order the action space uses. Where that block sits and what order it is in
# are both measured rather than assumed: an independent flood fill says how
# much room each move actually leads into, and the block of three consecutive
# channels that tracks it -- under any of the six orderings -- is the sensor.
#
# Whichever ordering is in place, the winning match scores 1.00 correlation
# and the runner-up 0.70, so there is a lot of room between the measurement's
# answer and its next guess.

LAYOUT_CONFIDENCE = 0.5


@pytest.fixture(scope="module")
def measured_layout():
  observations, truths = sample_states(states=400, seed=0)
  return free_space_layout(observations, truths, SnakeEnv)


@pytest.mark.issue(2)
def test_the_free_space_readings_can_be_found_at_all(measured_layout):
  """Guard on the measurement below: something in the observation tracks room.

  If no three consecutive channels move with the room available in any order,
  the free-space sensor is not reporting free space, and asking what order it
  is in has no answer.
  """
  start, permutation, score, _ = measured_layout
  assert score > LAYOUT_CONFIDENCE, (
      f"no three consecutive observation channels track how much room each "
      f"move leads into (best match {score:+.2f} at {start}:"
      f"{start + FREE_SPACE_COUNT} in order {permutation}, and "
      f"{LAYOUT_CONFIDENCE:+.2f} is the bar). The free-space sensors are not "
      "reporting free space at all.")


@pytest.mark.issue(2)
def test_the_free_space_readings_arrive_in_action_order(measured_layout):
  """Issue #2, stated as the sensors rather than as the symptom.

  "The three free-space readings are meant to arrive in that same order, so
  the first one describes what happens if you turn left."
  """
  start, permutation, score, runners_up = measured_layout
  detail = "\n".join(
      f"    {other_score:+.2f}  channels {other_start}:"
      f"{other_start + FREE_SPACE_COUNT} in order {other_permutation}"
      for other_score, other_start, other_permutation in runners_up)
  assert permutation == tuple(ACTIONS), (
      f"the free-space readings are in the wrong order. Measured at "
      f"{score:+.2f} correlation: {describe_layout(start, permutation)}.\n"
      "  They have to arrive in the same turn left / straight / turn right "
      "order the action space uses, so each reading lines up with the action "
      "it informs.\n"
      f"  next best matches:\n{detail}")


@pytest.fixture(scope="module")
def rule_scores(measured_layout):
  """Both hand-written players, scored without torch. About 7 seconds."""
  start, _, _, _ = measured_layout
  return score_rules(start, seeds=(0, 1, 2), episodes=100)


@pytest.mark.issue(2)
def test_food_plus_safety_is_the_strongest_player_again(rule_scores):
  """Issue #2's "Done when", as the comparison the README makes.

  `food+safety` is `greedy-food` plus a veto on moves into too little room,
  so with the sensors in order it has to come out ahead. Compared against
  `greedy-food` measured in the same run rather than against the README's
  41.12, so a submission that changed the game in some other legitimate way
  is still judged on the thing the issue is about.
  """
  safety_length, _ = rule_scores["food+safety"]
  greedy_length, _ = rule_scores["greedy-food"]
  assert safety_length > greedy_length, (
      f"'food+safety' reaches mean length {safety_length:.2f} against "
      f"'greedy-food' at {greedy_length:.2f}. It is the same player plus a "
      "veto on moving into too little room, so it cannot be the weaker of "
      f"the two unless the room it is reading is not the room it is moving "
      f"into.\n{rules_table(rule_scores)}")


@pytest.mark.issue(2)
def test_food_plus_safety_stops_dying_every_episode(rule_scores):
  """The other half of the same "Done when": about 15% died, not 100%.

  The bar is 50% rather than 15% on purpose -- the exact rate moves with any
  honest change to the game, and what the issue describes is a player that
  dies *every single time*.
  """
  _, safety_deaths = rule_scores["food+safety"]
  assert safety_deaths < 0.5, (
      f"'food+safety' still dies in {safety_deaths:.0%} of its episodes; the "
      "README has it at about 15%, and anything under 50% counts as fixed "
      f"here.\n{rules_table(rule_scores)}")


# --- issue #3: training teaches the snake to die -----------------------------
#
# REINFORCE pushes the policy toward whatever the return says was good. Get
# the direction backwards and it reliably searches for the worst behaviour
# available, which in Snake is crashing immediately -- so the test is not
# "does it train well" but "does training move the policy up or down".
#
# Everything is measured against a *freshly initialised policy on the same
# seed*, so there is no magic number to argue with.
#
# Two things set the bars below. **Deaths separate cleanly and returns do
# not**: every correct fix drops the death rate by at least half on every
# seed, while a correct-but-slower fix can still come out *behind* on return
# for an individual seed, because it learns to stop crashing before it learns
# to eat and starving costs the same -10 a crash does. So the return check
# runs on the mean of five seeds rather than three, and its bar sits between
# the two populations rather than just above the broken one.
#
# `docs/grading.md` on this branch has the measurements the bars came from.

TRAINING_EPISODES = 300
GROWTH_EPISODES = 600
TRAINING_SEEDS = (0, 1, 2, 3, 4)
EVAL_EPISODES = 30

# Sits about two points clear of both populations, so there is real room on
# either side of it.
RETURN_MARGIN = 1.5
# Well inside the drop every correct fix manages, with room left for one that
# only halves the death rate.
DEATH_MARGIN = 0.15
GROWTH_LENGTH = 6.0


@pytest.fixture(scope="module")
def training_run():
  """Trained against untrained on the same seeds. About 35 seconds."""
  import torch

  # Pinned for the same reason benchmark.py pins it: the thread count
  # reorders float reductions, and a graded number that moves with the
  # runner's core count is not a graded number.
  torch.set_num_threads(2)

  trained, baseline = [], []
  for seed in TRAINING_SEEDS:
    policy = train_policy(episodes=TRAINING_EPISODES, seed=seed)
    trained.append(score_policy(policy, seed, EVAL_EPISODES))
    baseline.append(score_policy(untrained_policy(seed), seed, EVAL_EPISODES))
  return trained, baseline


def _mean(runs, key):
  return float(np.mean([run[key] for run in runs]))


def _comparison(trained, baseline):
  rows = ["  seed   trained                          untrained",
          "  ----   ------------------------------   ------------------------------"]
  for seed, after, before in zip(TRAINING_SEEDS, trained, baseline):
    rows.append(
        f"  {seed:<4}   return {after['return']:8.2f}  died {after['death_rate']:4.2f}   "
        f"return {before['return']:8.2f}  died {before['death_rate']:4.2f}")
  return "\n".join(rows)


@pytest.mark.issue(3)
def test_training_stops_making_the_policy_worse_than_random(training_run):
  """Issue #3's headline symptom: "a snake taking random moves survives longer".

  The clearest single number in the issue is `deaths=1.00` -- it crashes in
  every episode, which an untrained policy does not manage. Training has to
  leave the snake dying *less* than it did before training, not more.
  """
  trained, baseline = training_run
  after = _mean(trained, "death_rate")
  before = _mean(baseline, "death_rate")
  assert after < before - DEATH_MARGIN, (
      f"after {TRAINING_EPISODES} episodes the policy dies in {after:.0%} of "
      f"its evaluation episodes; an untrained policy on the same seeds dies "
      f"in {before:.0%}. Training is moving the snake away from surviving, "
      "which is what an update pushed in the wrong direction does.\n"
      + _comparison(trained, baseline))


@pytest.mark.issue(3)
def test_training_raises_the_return_it_is_optimising(training_run):
  """Issue #3's other half: "eval_return never climbs".

  REINFORCE exists to push the return up. Measured against the same seed's
  untrained policy, so this asks whether training helped rather than whether
  it hit a particular number.

  Averaged across the seeds rather than required on each one. A correct fix
  that learns to survive before it learns to eat can lose return on an
  individual seed -- running the starvation clock out costs the same -10 that
  crashing does -- and failing that submission would be failing it for the
  order it learned things in.
  """
  trained, baseline = training_run
  after = _mean(trained, "return")
  before = _mean(baseline, "return")
  assert after > before + RETURN_MARGIN, (
      f"after {TRAINING_EPISODES} episodes eval_return is {after:+.2f}, "
      f"against {before:+.2f} for an untrained policy on the same seeds -- "
      f"an improvement of {after - before:+.2f}, where {RETURN_MARGIN:+.2f} "
      "is the bar. Training is not raising the quantity it is supposed to be "
      "maximising.\n" + _comparison(trained, baseline))


@pytest.mark.issue(3)
def test_a_longer_run_grows_the_snake():
  """Issue #3's "Done when", scaled to something CI can afford.

  The issue asks for `eval_length` in the high 20s after the shipped 2000
  episodes. That is a several-minute run per seed, so this takes the same
  measurement at 600 episodes, where a working trainer reaches about 12.8 and
  a broken one sits at the starting length of 3. The bar is 6 -- comfortably
  above "never grew", comfortably below "learned as fast as the reference" --
  because how *fast* a correct fix learns is not what issue #3 is about.
  """
  import torch

  torch.set_num_threads(2)
  policy = train_policy(episodes=GROWTH_EPISODES, seed=0)
  metrics = score_policy(policy, 0, 50)
  assert metrics["length"] > GROWTH_LENGTH, (
      f"after {GROWTH_EPISODES} episodes the snake reaches a mean length of "
      f"{metrics['length']:.2f} (it starts at 3, and the reference run "
      f"reaches about 12.8 by here). It is not learning to eat.\n"
      f"  return {metrics['return']:+.2f}   score {metrics['score']:.1f}   "
      f"died {metrics['death_rate']:.0%}   "
      f"steps/food {metrics['steps_per_food']:.1f}")

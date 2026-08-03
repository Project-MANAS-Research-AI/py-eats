"""Env invariants: the movement rules, the reward split, expiry and scoring.

These pin the *rules of the game*. The learning itself is stochastic and not
worth asserting on, but every number the policy is trained against is
deterministic and worth locking down -- a silent change to the reward is the
kind of bug that shows up three days later as "training got worse".
"""

import collections

import numpy as np
import pytest

from snake_rl.env import _DIRECTIONS, LEFT, RIGHT, STRAIGHT, TURN_LEFT, TURN_RIGHT, UP, SnakeEnv


def step_deltas(episodes, seed, **env_kwargs):
  """How much `_food_distance` moves per ordinary step, as a histogram.

  Eating steps and deaths are skipped: eating moves the food, so the two
  distances are measured against different targets, and a death has no
  "after" to compare. Neither is a step the shaping pays on, so neither
  belongs in the count.
  """
  deltas = collections.Counter()
  rng = np.random.default_rng(seed)
  env = SnakeEnv(seed=seed, **env_kwargs)
  for _ in range(episodes):
    env.reset()
    while True:
      before = env._food_distance()
      _, _, done, info = env.step(int(rng.integers(env.action_dim)))
      if done or info["ate"]:
        break
      deltas[env._food_distance() - before] += 1
  return deltas


def step_until(env, predicate, action=STRAIGHT, limit=1000):
  """Drive the env until `predicate(info)` holds; return that info."""
  for _ in range(limit):
    _, _, done, info = env.step(action)
    if predicate(info):
      return info
    if done:
      env.reset()
  raise AssertionError("predicate never held")


def place_food_ahead(env, gap=1):
  """Put the food `gap` cells straight ahead of the head, whatever the
  heading -- so a test can turn first and still line the food up."""
  d_row, d_col = _DIRECTIONS[env.direction]
  head_row, head_col = env.snake[0]
  env.food = (head_row + d_row * gap, head_col + d_col * gap)
  return env.food


class TestMovement:
  def test_starts_at_length_three_moving_right(self):
    env = SnakeEnv(seed=0)
    assert len(env.snake) == 3
    assert env.direction == RIGHT

  def test_turns_are_relative_to_heading(self):
    env = SnakeEnv(seed=0)
    env.step(TURN_LEFT)
    assert env.direction == UP
    env.step(TURN_RIGHT)
    assert env.direction == RIGHT

  def test_no_single_action_reverses_into_the_neck(self):
    """The whole reason actions are relative -- there is no 180 to take."""
    for action in (TURN_LEFT, STRAIGHT, TURN_RIGHT):
      env = SnakeEnv(seed=0)
      neck = env.snake[1]
      env.step(action)
      assert env.snake[0] != neck

  def test_moving_into_the_vacating_tail_is_not_a_crash(self):
    env = SnakeEnv(seed=0)
    # A 2x2 ring with the head aimed squarely at its own tail cell, which
    # vacates as the body advances -- so this must be legal. The assert on
    # the setup is not decoration: an earlier version of this test aimed the
    # head at an empty square and passed no matter what the rule was.
    env.snake = [(5, 5), (6, 5), (6, 4), (5, 4)]
    env.direction = LEFT
    env.food = (0, 11)
    assert env.cell_ahead(env.direction) == env.snake[-1]
    _, _, done, info = env.step(STRAIGHT)
    assert not info["died"] and not done

  def test_running_into_the_body_is_still_a_crash(self):
    """The other half of the tail rule: only the last segment vacates."""
    env = SnakeEnv(seed=0)
    env.snake = [(5, 5), (5, 4), (6, 4), (6, 5), (7, 5), (7, 4)]
    env.direction = LEFT
    env.food = (0, 11)
    _, _, _, info = env.step(TURN_LEFT)  # LEFT + left turn = DOWN, into (6, 5)
    assert info["died"]

  def test_wall_kills(self):
    env = SnakeEnv(seed=0, grid_size=6)
    env.snake = [(0, 3), (0, 2), (0, 1)]
    env.direction = RIGHT
    env.food = (5, 5)
    info = step_until(env, lambda i: i["died"] or i["ate"])
    assert info["died"]


class TestRewardSplit:
  """Defaults must reproduce the classic shaped reward exactly."""

  def test_eating_pays_what_the_scoreboard_paid(self):
    """Reward and score agree on a meal, so speed is something the optimizer
    can actually see. A flat +10 priced a two-step meal and a forty-step one
    the same."""
    env = SnakeEnv(seed=0)
    place_food_ahead(env)
    _, reward, _, info = env.step(STRAIGHT)
    assert info["ate"]
    assert 10.0 <= info["points"] <= 20.0
    assert reward == pytest.approx(info["points"])
    assert info["food_reward"] == pytest.approx(info["points"])
    assert info["survival_reward"] == pytest.approx(0.0)

  def test_dying_costs_ten_and_is_all_survival_term(self):
    env = SnakeEnv(seed=0, grid_size=6)
    env.snake = [(0, 3), (0, 2), (0, 1)]
    env.direction = RIGHT
    env.food = (5, 5)
    _, reward, _, info = env.step(TURN_LEFT)  # into the top wall
    assert info["died"]
    assert reward == pytest.approx(-10.0)
    assert info["survival_reward"] == pytest.approx(-10.0)
    assert info["food_reward"] == pytest.approx(0.0)

  def test_approaching_and_retreating_are_symmetric_around_step_cost(self):
    env = SnakeEnv(seed=0)
    place_food_ahead(env, gap=5)
    _, closer, _, _ = env.step(STRAIGHT)
    env.reset()
    place_food_ahead(env, gap=5)
    _, away, _, _ = env.step(TURN_LEFT)
    assert closer == pytest.approx(0.1 - 0.01)
    assert away == pytest.approx(-0.1 - 0.01)
    # The shaping cancels over a there-and-back pair; only the standing cost
    # of the two steps is left, which is what stops loop-riding being free.
    assert closer + away == pytest.approx(-2 * 0.01)

  def test_distance_never_moves_by_more_than_one_cell_a_step(self):
    """The assumption the pair above is a special case of.

    A flat `+/-0.1` on the *sign* of the change only cancels around a closed
    loop while every step moves the distance by exactly one. Let a step jump
    by three and the loop pays out: one `-0.1` against three `+0.1`s. That is
    a property of the distance measure rather than of the reward code, which
    is why it is checked over real play instead of a hand-built pair. A BFS
    distance through open cells breaks exactly this, which is why that
    measure was tried and removed for this reason.
    """
    deltas = step_deltas(episodes=50, seed=0)
    assert deltas, "no ordinary steps were sampled"
    assert set(deltas) == {-1, 1}

  def test_starving_costs_the_same_as_crashing(self):
    """Both are the snake not surviving. Pricing them differently teaches it
    to prefer one kind of dying, which is how it learned to park."""
    env = SnakeEnv(seed=0)
    env.food = (11, 11)  # far away, and never approached
    for _ in range(env.starvation_limit + 2):
      _, _, done, info = env.step(TURN_LEFT)
      if done:
        break
    assert info["starved"] and not info["died"]
    assert info["survival_reward"] == pytest.approx(-10.0)



class TestObservation:
  def test_dimensions_match_the_advertised_size(self):
    env = SnakeEnv(seed=0)
    assert env.observation_dim == 18
    assert env.reset().shape == (18,)

  def test_the_last_value_is_the_food_clock(self):
    """The policy is paid on how fast it reaches food, so it has to be able
    to see how much of that clock is left."""
    env = SnakeEnv(seed=0)
    place_food_ahead(env, gap=8)
    observation, _, _, _ = env.step(TURN_LEFT)
    assert observation[-1] == pytest.approx(env.food_freshness())
    assert observation[-1] < 1.0

  def test_food_direction_bits_track_the_food(self):
    env = SnakeEnv(seed=0)
    head_row, head_col = env.snake[0]
    env.food = (head_row - 2, head_col + 3)
    observation = env._observation()
    food_up, food_down, food_left, food_right = observation[7:11]
    assert (food_up, food_down, food_left, food_right) == (1.0, 0.0, 0.0, 1.0)

  def test_observations_are_float32_and_finite(self):
    env = SnakeEnv(seed=0)
    observation = env.reset()
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))


class TestScoring:
  def test_score_starts_at_zero_and_rises_only_on_eating(self):
    env = SnakeEnv(seed=0)
    assert env.score == 0
    env.step(TURN_LEFT)
    assert env.score == 0
    place_food_ahead(env)
    _, _, _, info = env.step(STRAIGHT)
    assert info["ate"] and env.score > 0
    assert env.foods_eaten == 1

  def test_fresh_food_pays_double_and_stale_food_pays_base(self):
    env = SnakeEnv(seed=0)
    place_food_ahead(env)
    _, _, _, info = env.step(STRAIGHT)
    # Reached immediately: full clock left, so base + full bonus.
    assert info["points"] == 20
    # Now dawdle until the starvation clock has almost run out.
    place_food_ahead(env)
    env.steps_since_food = env.starvation_limit - 1
    _, _, _, info = env.step(STRAIGHT)
    assert info["points"] == 10

  def test_a_direct_route_pays_full_bonus_at_any_distance(self):
    """The bonus is scored against the food's own spawn distance, so a near
    meal and a far one are the same task if both are walked straight at.
    A fixed clock would pay the far one less for no fault of the snake."""
    # Head starts at column 6 on a 12-wide board, so 5 is the longest
    # straight run that stays on it.
    for gap in (1, 3, 5):
      env = SnakeEnv(seed=0)
      place_food_ahead(env, gap=gap)
      for _ in range(gap):
        _, _, _, info = env.step(STRAIGHT)
      assert info["ate"], f"gap {gap} was not reached"
      assert info["points"] == 20, f"gap {gap} paid {info['points']}"

  def test_dawdling_pays_less_than_walking_straight(self):
    """Same food, same distance, twice the steps -- half the bonus."""
    env = SnakeEnv(seed=0)
    place_food_ahead(env, gap=2)
    # Pretend two steps of detour happened first. The counter is not
    # incremented on the eating step itself, so this lands at 4 when the
    # meal is scored: spawn distance 2, taken in 4 -> efficiency 0.5.
    env.steps_since_food = 3
    env.step(STRAIGHT)
    _, _, _, info = env.step(STRAIGHT)
    assert info["ate"]
    assert info["points"] == 15

  def test_score_and_reward_agree_on_a_meal(self):
    """A meal pays the scoreboard and the reward the same number.

    Let the two drift apart -- a route bonus on the scoreboard against a flat
    payout in the reward -- and the snake is graded on a speed nothing in
    training ever asked it for."""
    env = SnakeEnv(seed=0)
    place_food_ahead(env)
    _, reward, _, info = env.step(STRAIGHT)
    assert env.score == info["points"]
    assert reward == pytest.approx(info["points"])

  def test_the_last_meal_on_a_full_board_pays_like_any_other(self):
    """The meal most likely to be given a special case, and it gets none.

    A flat bonus for filling the board, paid in reward while the scoreboard
    still recorded the meal's usual 10 to 20, would be an exception nothing
    documents and only the final meal of a perfect game could ever reach.
    """
    env = SnakeEnv(seed=0, grid_size=4)
    # A boustrophedon path covering 15 of the 16 cells, head one step from
    # the last empty one, so eating it fills the board exactly.
    env.snake = [(3, 1), (3, 2), (3, 3), (2, 3), (2, 2), (2, 1), (2, 0),
                 (1, 0), (1, 1), (1, 2), (1, 3), (0, 3), (0, 2), (0, 1),
                 (0, 0)]
    env.direction = LEFT
    env.food = (3, 0)
    _, reward, done, info = env.step(STRAIGHT)
    assert info["ate"] and info["filled_board"] and done
    assert info["length"] == env.grid_size ** 2
    assert reward == pytest.approx(info["points"])
    assert env.score == info["points"]

  def test_reset_clears_the_score(self):
    env = SnakeEnv(seed=0)
    place_food_ahead(env)
    env.step(STRAIGHT)
    assert env.score > 0
    env.reset()
    assert env.score == 0 and env.foods_eaten == 0


class TestEpisodeEnd:
  def test_starvation_clock_ends_the_episode(self):
    """Isolated from crashing: one step short of the limit, moving into open
    board, so hunger is the only thing that can end this.

    An earlier version of this test circled until *something* happened and
    accepted `starved or died`, which a snake that crashed satisfied without
    the clock ever firing.
    """
    env = SnakeEnv(seed=0, grid_size=12, horizon=10_000)
    env.food = (0, 11)
    env.steps_since_food = env.starvation_limit - 1
    _, _, done, info = env.step(STRAIGHT)
    assert not info["died"]
    assert done and info["starved"]

  def test_eating_resets_the_starvation_clock(self):
    env = SnakeEnv(seed=0, grid_size=12, horizon=10_000)
    env.steps_since_food = env.starvation_limit - 1
    env.food = env.cell_ahead(env.direction)
    _, _, done, info = env.step(STRAIGHT)
    assert info["ate"] and not done

  def test_horizon_ends_the_episode(self):
    env = SnakeEnv(seed=0, horizon=5)
    for _ in range(4):
      assert not env.step(STRAIGHT)[2]
    assert env.step(STRAIGHT)[2]

  def test_step_returns_four_values(self):
    env = SnakeEnv(seed=0)
    result = env.step(STRAIGHT)
    assert len(result) == 4
    _observation, reward, done, info = result
    assert isinstance(reward, float) and isinstance(done, bool)
    assert set(info) >= {"ate", "died", "starved", "score",
                         "points", "length", "food_reward", "survival_reward"}


class TestDeterminism:
  def test_same_seed_gives_the_same_game(self):
    def play(seed):
      env = SnakeEnv(seed=seed)
      rng = np.random.default_rng(0)
      trace = []
      for _ in range(60):
        _, reward, done, _ = env.step(int(rng.integers(3)))
        trace.append((reward, env.food, env.score))
        if done:
          env.reset()
      return trace

    assert play(3) == play(3)
    assert play(3) != play(4)

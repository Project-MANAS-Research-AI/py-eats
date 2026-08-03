"""A grid + action-confidence viewer for the toy Snake environment.

Like the sibling toy_quad viewer, this is a schematic rather
than a game renderer: the point is to make the policy's *decision*
visible, not just the board.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgb

from snake_rl.env import DOWN, LEFT, RIGHT, STRAIGHT, TURN_LEFT, TURN_RIGHT, UP, SnakeEnv
from snake_rl.models import SnakePolicy

# Actions are relative to the heading, so the labels are turns.
ACTION_LABELS = ("TURN LEFT", "STRAIGHT", "TURN RIGHT")


class SnakeViewer:
  def __init__(self, random_actions=False, checkpoint=None, seed=0,
               stop_at_death=False, greedy=False, horizon=800):
    # A checkpoint records the rules it was trained under, and the env is
    # rebuilt from them, so a policy is always replayed on the game it
    # learned rather than on whatever the defaults happen to be today.
    saved = None
    if checkpoint:
      saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    env_kwargs = dict(saved["env_kwargs"]) if saved else {}
    env_kwargs["horizon"] = horizon  # --horizon is the viewer's to set
    self.env = SnakeEnv(seed=seed, **env_kwargs)
    self.observation = self.env.reset()
    self.random_actions = random_actions
    # Sampling from the policy (rather than always taking the argmax) is
    # what makes two runs of the same checkpoint differ: different route,
    # different food, different episode length. Its own generator, so
    # drawing actions never shifts the env's food sequence.
    self.greedy = greedy
    self.action_rng = np.random.default_rng(seed + 1)
    # When saving a GIF we play a single episode through to its end; the
    # live window instead loops forever so there is always something to
    # watch.
    self.stop_at_death = stop_at_death
    self.finished = False
    self.outcome = None
    self.freeze_tick = 0
    self.policy = None
    if saved:
      self.policy = SnakePolicy(
        observation_dim=saved.get("observation_dim",
                                  self.env.observation_dim),
        hidden_dim=saved.get("hidden_dim", 64))
      self.policy.load_state_dict(saved["policy_state_dict"])
      self.policy.eval()
    self.last_probabilities = np.full(3, 1 / 3, dtype=np.float32)

    self.figure, (self.grid_ax, self.prob_ax) = plt.subplots(
        1, 2, figsize=(10, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    title = ("Trained Snake policy: action confidence made visible"
             if self.policy else "Toy Snake baseline: action confidence made visible")
    self.figure.suptitle(title, fontsize=14, fontweight="bold")
    self.figure.subplots_adjust(top=0.82, wspace=0.35)

    grid_size = self.env.grid_size
    self.grid_ax.set_title("Board (dark head, fading tail, red food)", pad=12)
    self.grid_ax.set_xticks([])
    self.grid_ax.set_yticks([])
    self.image_artist = self.grid_ax.imshow(np.ones((grid_size, grid_size, 3)),
                                            interpolation="nearest")
    self.status = self.grid_ax.text(0.0, -0.06, "", transform=self.grid_ax.transAxes,
                                    fontsize=10, color="#1e293b", va="top")

    self.prob_ax.set_title("Action confidence", pad=12)
    self.bars = self.prob_ax.bar(ACTION_LABELS, self.last_probabilities, color="#4f83cc")
    self.prob_ax.set_ylim(0, 1.0)
    self.prob_ax.set_ylabel("probability")

  def _greedy_action(self):
    """Hand-written demo: turn toward the food, ignoring danger."""
    head_row, head_col = self.env.snake[0]
    food_row, food_col = self.env.food
    if food_row < head_row:
      desired = UP
    elif food_row > head_row:
      desired = DOWN
    elif food_col < head_col:
      desired = LEFT
    elif food_col > head_col:
      desired = RIGHT
    else:
      desired = self.env.direction
    # Turn the absolute target into a relative turn. A reversal isn't
    # expressible in one action, so fall back to going straight.
    return {0: STRAIGHT, 1: TURN_RIGHT, 3: TURN_LEFT}.get(
        (desired - self.env.direction) % 4, STRAIGHT)

  def _decide(self):
    if self.policy is not None:
      with torch.no_grad():
        observation = torch.as_tensor(self.observation).unsqueeze(0)
        logits = self.policy.net(observation).squeeze(0)
        probabilities = torch.softmax(logits, dim=-1).numpy()
      if self.greedy:
        return int(np.argmax(probabilities)), probabilities
      # Play the policy the way it was trained -- by sampling. Argmax
      # playback is a strictly worse demo: it locks the snake onto one
      # route per seed, and a policy with a favourite safe cycle will ride
      # it to the starvation clock every single time.
      action = int(self.action_rng.choice(len(probabilities), p=probabilities))
      return action, probabilities
    if self.random_actions:
      # action_rng, not env.rng: drawing actions from the env's generator
      # would shift its food sequence, so the same seed would lay food out
      # differently under --random than under a policy.
      action = int(self.action_rng.integers(3))
      return action, np.full(3, 1 / 3, dtype=np.float32)
    action = self._greedy_action()
    probabilities = np.zeros(3, dtype=np.float32)
    probabilities[action] = 1.0
    return action, probabilities

  def _draw(self):
    grid_size = self.env.grid_size
    image = np.full((grid_size, grid_size, 3), to_rgb("#e2e8f0"))
    food_row, food_col = self.env.food
    # Food fades from red toward the board colour as the starvation clock
    # runs down, so the deadline the policy is shown -- the one that ends the
    # episode if it dawdles -- is visible on screen too.
    freshness = self.env.food_freshness()
    image[food_row, food_col] = (
      np.array(to_rgb("#ef4444")) * freshness
      + np.array(to_rgb("#e2e8f0")) * (1.0 - freshness))

    body_length = len(self.env.snake)
    for index, (row, col) in enumerate(self.env.snake):
      fade = index / max(1, body_length - 1)
      image[row, col] = plt.cm.viridis(0.85 - 0.55 * fade)[:3]
    head_row, head_col = self.env.snake[-1]
    # Flash the head once the episode is over. Besides reading as "you
    # died", it keeps the held frames from being pixel-identical -- the GIF
    # writer collapses consecutive identical frames, which would silently
    # drop the pause entirely.
    flashing = self.finished and self.freeze_tick % 2 == 1
    image[head_row, head_col] = to_rgb("#ffffff" if flashing else "#172554")
    self.image_artist.set_data(image)

    for bar, probability in zip(self.bars, self.last_probabilities):
      bar.set_height(probability)

    status = (f"length {body_length}   score {self.env.score}   "
              f"step {self.env.step_count}")
    remaining = self.env.starvation_limit - self.env.steps_since_food
    # What this meal would pay if taken now, so the route bonus is legible
    # rather than something that only shows up in the final score.
    worth = round(10 * (1.0 + self.env.route_efficiency()))
    status += f"   starve in {max(0, remaining)}   meal worth {worth}"
    if self.outcome:
      status += f"   --  {self.outcome}"
    self.status.set_text(status)

  def update(self, _):
    if self.finished:
      # Hold the final board so the GIF does not cut off the instant the
      # snake dies.
      self.freeze_tick += 1
      self._draw()
      return [self.image_artist, *self.bars, self.status]

    action, probabilities = self._decide()
    self.last_probabilities = probabilities
    self.observation, _, done, info = self.env.step(action)
    if done:
      if self.stop_at_death:
        self.finished = True
        # Read the ending off `info` rather than off the reward: crashing
        # and starving pay exactly the same -10, so the reward alone cannot
        # tell the two apart.
        if info["died"]:
          self.outcome = "CRASHED"
        elif info["filled_board"]:
          self.outcome = "FILLED THE BOARD"
        elif info["starved"]:
          self.outcome = "STARVED"
        else:
          self.outcome = "OUT OF TIME"
      else:
        self.observation = self.env.reset()
    self._draw()
    return [self.image_artist, *self.bars, self.status]

  def episode_frames(self, limit, freeze=8):
    """Frame numbers for exactly one episode, plus a short freeze at the end.

    FuncAnimation stops when this generator does, so the GIF ends up as
    long as the episode was rather than a fixed frame count.
    """
    held = 0
    for index in range(limit):
      if self.finished:
        if held >= freeze:
          return
        held += 1
      yield index


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--random", action="store_true",
                      help="use random actions instead of the greedy demo heuristic")
  parser.add_argument("--checkpoint", metavar="PATH",
                      help="play back a trained checkpoint, sampling from its "
                           "action distribution (see --greedy)")
  parser.add_argument("--seed", type=int,
                      help="fix the run (food layout and action sampling); "
                           "omit for a fresh random seed every run")
  parser.add_argument("--greedy", action="store_true",
                      help="take the argmax action instead of sampling, which "
                           "makes a given seed replay one identical route")
  parser.add_argument("--save", metavar="PATH",
                      help="save a GIF instead of opening a GUI window")
  parser.add_argument("--horizon", type=int, default=800,
                      help="hard step limit on the episode; raise it to let a "
                           "surviving snake keep growing instead of being cut "
                           "off by the clock (default: 800)")
  parser.add_argument("--frames", type=int,
                      help="safety cap on GIF length; the episode normally ends "
                           "first (default: the horizon plus the end freeze)")
  args = parser.parse_args()
  # Defaulting the cap to the horizon exactly would eat the end freeze on a
  # full-length episode, so leave the freeze its own room.
  frames = args.frames if args.frames is not None else args.horizon + 20
  # A run with no --seed picks its own, and reports it, so an interesting
  # episode can still be reproduced afterwards.
  seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2 ** 32))
  viewer = SnakeViewer(random_actions=args.random, checkpoint=args.checkpoint,
                       seed=seed, stop_at_death=bool(args.save),
                       greedy=args.greedy, horizon=args.horizon)
  animation_kwargs = {"interval": 120, "blit": False, "cache_frame_data": False}
  if args.save:
    animation_kwargs["frames"] = viewer.episode_frames(frames)
    animation_kwargs["save_count"] = frames
    # Without an init_func, FuncAnimation initialises the figure by pulling
    # the first item off the frame generator -- which steps the env and
    # draws it, but is never handed to the writer, silently dropping the
    # opening move from the GIF.
    animation_kwargs["init_func"] = lambda: [viewer.image_artist, *viewer.bars,
                                             viewer.status]
  animation = FuncAnimation(viewer.figure, viewer.update, **animation_kwargs)
  if args.save:
    # dpi below the figure's own: a full episode is a few hundred frames, and
    # at the default 100 the GIF lands in the tens of megabytes for a board
    # that is a grid of flat colour blocks and three bars.
    animation.save(args.save, writer="pillow", fps=8,
                   savefig_kwargs={"facecolor": "white"}, dpi=72)
    print(f"Saved animation to {args.save} "
          f"({viewer.outcome or 'stopped at frame cap'} "
          f"at length {len(viewer.env.snake)}, "
          f"score {viewer.env.score}, "
          f"step {viewer.env.step_count}, seed {seed})")
  else:
    backend = plt.get_backend().lower()
    if "agg" in backend:
      raise RuntimeError(
        "No interactive display is available. Use --save snake.gif "
        "to render a GIF instead.")
    plt.show()
  return animation


if __name__ == "__main__":
  main()

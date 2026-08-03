"""Train a Snake MLP policy with REINFORCE on the shaped eat/die reward."""

import argparse

import matplotlib

matplotlib.use("Agg")  # this script only ever saves plots, never shows one
import matplotlib.pyplot as plt
import numpy as np
import torch

from snake_rl.env import SnakeEnv
from snake_rl.models import SnakePolicy


def compute_returns(rewards, gamma):
  returns = np.zeros(len(rewards), dtype=np.float32)
  running = 0.0
  for t in reversed(range(len(rewards))):
    running = rewards[t] + gamma * running
    returns[t] = running
  return returns


def rollout(env, policy):
  """Play one episode, returning the per-step tensors REINFORCE needs."""
  observation, done = env.reset(), False
  log_probs, entropies, rewards = [], [], []
  while not done:
    obs_tensor = torch.as_tensor(observation).unsqueeze(0)
    action, log_prob, entropy = policy.sample(obs_tensor)
    observation, reward, done, _ = env.step(action.item())
    log_probs.append(log_prob.squeeze(0))
    entropies.append(entropy.squeeze(0))
    rewards.append(reward)
  return log_probs, entropies, rewards


def evaluate(policy, seed, episodes=8, env_kwargs=None):
  """Score the policy the same way the viewer plays it back: by sampling.

  Evaluating with argmax measures a policy the viewer never runs, and it
  hides exactly the failure we care about -- a deterministic argmax route
  can sit in a safe cycle forever, which reads as a fine "eval_length" while
  being the degenerate behaviour.

  `env_kwargs` is the *same* configuration train() built its env from, so a
  non-default horizon or step cost cannot leave the policy scored on a
  different game than the one it is learning.
  """
  env = SnakeEnv(seed=seed + 10_000, **(env_kwargs or {}))
  returns, final_lengths, scores, foods, steps, deaths = [], [], [], [], [], []
  policy.eval()
  with torch.no_grad():
    for _ in range(episodes):
      observation, done, total_reward = env.reset(), False, 0.0
      info = {}
      while not done:
        observation_tensor = torch.as_tensor(observation).unsqueeze(0)
        action, _, _ = policy.sample(observation_tensor)
        observation, reward, done, info = env.step(action.item())
        total_reward += reward
      returns.append(total_reward)
      final_lengths.append(len(env.snake))
      scores.append(env.score)
      foods.append(env.foods_eaten)
      steps.append(env.step_count)
      deaths.append(float(info.get("died", False)))
  policy.train()
  total_foods = float(np.sum(foods))
  return {
    "return": float(np.mean(returns)),
    "length": float(np.mean(final_lengths)),
    "score": float(np.mean(scores)),
    "foods": float(np.mean(foods)),
    # How long the snake takes per meal -- the number that moves when food
    # is put on a clock, in a way average length can hide.
    "steps_per_food": float(np.sum(steps) / total_foods) if total_foods
                      else float("inf"),
    # What fraction of episodes ended in a crash rather than on a clock.
    # This is the death-avoidance axis, made measurable.
    "death_rate": float(np.mean(deaths)),
  }


def save_training_curve(path, eval_episodes, eval_returns, eval_lengths,
                        episodes_per_eval=8, eval_every=25):
  """Plot the eval metrics collected during train() and save them as a PNG
  -- a durable artifact of a run, versus the printed log lines which scroll
  away.

  The eval cadence is in the title because it is the thing that makes two
  curves incomparable, and it is invisible once the run has scrolled away.
  """
  figure, return_axis = plt.subplots(figsize=(7, 4))
  return_axis.plot(eval_episodes, eval_returns, color="#4f83cc", marker="o",
                   markersize=3, label="eval_return")
  return_axis.set_xlabel("episode")
  return_axis.set_ylabel("eval_return", color="#4f83cc")
  return_axis.tick_params(axis="y", labelcolor="#4f83cc")

  length_axis = return_axis.twinx()
  length_axis.plot(eval_episodes, eval_lengths, color="#16a34a", marker="o",
                   markersize=3, label="eval_length")
  length_axis.set_ylabel("eval_length", color="#16a34a")
  length_axis.tick_params(axis="y", labelcolor="#16a34a")

  figure.suptitle(f"Training progress (sampled eval, {episodes_per_eval} "
                  f"episodes every {eval_every})")
  figure.tight_layout()
  figure.savefig(path, dpi=120)
  plt.close(figure)


def train(episodes, seed, gamma, entropy_bonus, checkpoint, plot,
          lr=1e-3, hidden_dim=64, eval_every=25, eval_episodes=8,
          horizon=800, quiet=False, grid_size=12, step_cost=0.01,
          entropy_final=None):
  torch.manual_seed(seed)
  # None means "do not anneal": the weight stays at entropy_bonus all run.
  entropy_final = entropy_bonus if entropy_final is None else entropy_final
  # One dict describing the game, used to build the training env, the eval
  # env, and (via the checkpoint) the viewer's env. Three places that must
  # agree about the rules, fed from a single source.
  env_kwargs = {"grid_size": grid_size, "horizon": horizon,
                "step_cost": step_cost}
  env = SnakeEnv(seed=seed, **env_kwargs)
  policy = SnakePolicy(observation_dim=env.observation_dim,
                       hidden_dim=hidden_dim)
  optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
  eval_points, eval_returns, eval_lengths = [], [], []

  for episode in range(1, episodes + 1):
    # REINFORCE has no bootstrapping, so a complete episode is the smallest
    # unit carrying an unbiased return, and one of them is the update.
    log_probs, entropies, rewards = rollout(env, policy)
    returns = torch.as_tensor(compute_returns(rewards, gamma))
    # Standardized within the episode so the learning rate means the same
    # thing whether the snake ate twice or twenty times. unbiased=False:
    # very short episodes happen, and the Bessel-corrected std of a single
    # sample is 0/0 = nan.
    returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-6)

    # Subtracting the entropy keeps the action distribution from sharpening
    # into a one-hot policy, so playback stays varied and the snake can
    # still break out of a comfortable cycle.
    #
    # The weight is annealed straight-line from `entropy_bonus` to
    # `entropy_final` across the run: exploration is what an untrained snake
    # needs and what a trained one is held back by. Holding it at the
    # starting value to the last episode ships a policy that is still
    # deliberately playing badly some of the time.
    bonus = entropy_bonus + (entropy_final - entropy_bonus) * (
        (episode - 1) / max(1, episodes - 1))
    entropy_term = torch.stack(entropies).mean()
    policy_loss = ((torch.stack(log_probs) * returns).mean()
                   - bonus * entropy_term)
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    if episode == 1 or episode % eval_every == 0:
      metrics = evaluate(policy, seed, eval_episodes, env_kwargs)
      eval_points.append(episode)
      eval_returns.append(metrics["return"])
      eval_lengths.append(metrics["length"])
      if not quiet:
        print(f"episode={episode:4d} return={sum(rewards):7.2f} "
              f"eval_return={metrics['return']:7.2f} "
              f"eval_length={metrics['length']:5.2f} "
              f"eval_score={metrics['score']:6.1f} "
              f"steps/food={metrics['steps_per_food']:5.1f} "
              f"deaths={metrics['death_rate']:4.2f} "
              f"entropy={entropy_term.item():4.2f} w={bonus:5.3f}")

  if checkpoint:
    # env_kwargs is what the viewer rebuilds the game from; train_kwargs is
    # provenance only, nothing reads it back. It is recorded so a checkpoint
    # describes the run that produced it and not just the rules it was
    # played under.
    train_kwargs = {"gamma": gamma, "lr": lr, "entropy_bonus": entropy_bonus,
                    "entropy_final": entropy_final}
    torch.save({"policy_state_dict": policy.state_dict(), "seed": seed,
                "episodes": episodes, "hidden_dim": hidden_dim,
                "observation_dim": env.observation_dim,
                "env_kwargs": env_kwargs,
                "train_kwargs": train_kwargs}, checkpoint)
    if not quiet:
      print(f"Saved policy checkpoint to {checkpoint}")

  if plot:
    save_training_curve(plot, eval_points, eval_returns, eval_lengths,
                        eval_episodes, eval_every)
    if not quiet:
      print(f"Saved training curve to {plot}")
  return policy


def main():
  """The `py-eats-train` command.

  A function rather than a bare `if __name__ == "__main__"` block so that
  `pyproject.toml` can point a console script at it. Every default below is
  load-bearing: `py-eats-train` with no arguments is the configuration the
  README's numbers are measured on.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--episodes", type=int, default=2000,
                      help="REINFORCE is high-variance at this scale, so the "
                           "curve is still climbing when it stops; this is "
                           "the point where it has mostly levelled off and a "
                           "run still finishes in a couple of minutes")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--gamma", type=float, default=0.95,
                      help="discount: how far ahead an action is held "
                           "responsible, about 1/(1-gamma) steps. The "
                           "textbook 0.99 is a ~100-step window on a task "
                           "that pays out every ~16; sweep it with "
                           "'benchmark.py --preset gamma'")
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--hidden-dim", type=int, default=64)
  parser.add_argument("--eval-every", type=int, default=25)
  parser.add_argument("--eval-episodes", type=int, default=8)
  parser.add_argument("--horizon", type=int, default=800,
                      help="max steps per training episode. Real Snake has "
                           "no clock; this exists only so episodes end. Set "
                           "it low and it truncates the late game the snake "
                           "has just earned -- above ~600 it stops binding "
                           "and every episode ends by crashing or starving")
  parser.add_argument("--grid-size", type=int, default=12)
  parser.add_argument("--step-cost", type=float, default=0.01,
                      help="flat cost charged for every non-eating step, so "
                           "riding a safe loop is not free")
  parser.add_argument("--entropy-bonus", type=float, default=0.03,
                      help="weight on the entropy term that keeps the policy "
                           "stochastic, guarding against a collapse into one "
                           "fixed route")
  parser.add_argument("--entropy-final", type=float, default=None,
                      help="entropy weight at the last episode; the weight "
                           "moves straight-line from --entropy-bonus to this "
                           "across the run. Omit to hold it constant, which "
                           "is what ships. The idea is that exploration is "
                           "what an untrained snake needs and what a trained "
                           "one is held back by; it is off by default because "
                           "that is untested -- 'benchmark.py --preset "
                           "entropy' is the experiment")
  parser.add_argument("--checkpoint", default="trained_policy.pt",
                      help="path for the trained policy checkpoint")
  parser.add_argument("--plot", default="training_curve.png",
                      help="path for the eval-metrics plot PNG (pass an "
                           "empty string to skip saving one)")
  parser.add_argument("--quiet", action="store_true")
  train(**vars(parser.parse_args()))


if __name__ == "__main__":
  main()

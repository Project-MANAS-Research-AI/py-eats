"""Small MLP categorical policy for the Snake environment."""

from torch import nn
from torch.distributions import Categorical


class SnakePolicy(nn.Module):
  """Two-layer MLP policy; same trunk as toy_quad's Policy, discrete head.

  Where toy_quad's Policy outputs the mean of a Gaussian over continuous leg
  commands, this outputs logits over 3 discrete moves -- Snake's action space
  is inherently discrete, so the natural fit is a Categorical head rather
  than tanh-squashed Gaussian sampling.
  """

  def __init__(self, observation_dim=18, action_dim=3, hidden_dim=64):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(observation_dim, hidden_dim), nn.Tanh(),
      nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
      nn.Linear(hidden_dim, action_dim))

  def sample(self, observation):
    distribution = Categorical(logits=self.net(observation))
    # Categorical has no rsample(): discrete distributions aren't
    # reparameterizable, so .sample() is the only option -- and log_prob
    # already gives REINFORCE the score-function gradient it needs.
    action = distribution.sample()
    # The entropy comes back with the action so the trainer can pay a bonus
    # for keeping the distribution spread out. A REINFORCE policy that
    # collapses to one-hot logits plays a single fixed route forever, which
    # in Snake means happily circling a safe loop until it starves.
    return action, distribution.log_prob(action), distribution.entropy()

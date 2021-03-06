import torch
import torch.nn as nn
from .util import positionalEncoding
from .modules import TopDown, BottomUp, Attention

class GLOM(nn.Module):
  def __init__(self,
      levels=5,
      featureDim=64,
      inputChannels=3,
      inputResolution=256,
      patchSize=8,
      positionalCount=6):
    super().__init__()
    self.edge = inputResolution // patchSize
    self.featureDim = featureDim
    self.levels = levels
    # positional encoding dimension = 2 (x, y) * 2 (sin, cos) * count
    self.positionalDim = positionalCount * 2 * 2

    self.topDown = TopDown(levels, featureDim, self.positionalDim)
    self.bottomUp = BottomUp(levels, featureDim)
    self.attention = Attention(levels, featureDim)
    self.inputConv = nn.Conv2d(
      inputChannels,
      featureDim,
      kernel_size=patchSize,
      stride=patchSize)

    # positional embedding for the image coordinates
    self.register_buffer('positional', positionalEncoding(self.edge, positionalCount))

  # create the initial state tensor from an image
  def initState(self, batch):
    features = self.inputConv(batch)
    state = torch.zeros((batch.shape[0], self.levels + 1, self.featureDim, self.edge, self.edge), device=batch.device)
    state[:, 0] = features
    return state

  def step(self, state):
    # run top-down network
    topDown = self.topDown(state, self.positional)
    # run bottom-up network
    bottomUp = self.bottomUp(state)
    # run attention
    attn = self.attention(state)
    return 0.25 * (state + attn + bottomUp + topDown)

  def forward(self, batch, iters):
    state = self.initState(batch)
    for i in range(iters):
      state = self.step(state)
    return state

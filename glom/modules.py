import torch
import torch.nn as nn

class BottomUp(nn.Module):
  def __init__(self, levels, featureDim):
    super().__init__()
    self.levels = levels
    self.featureDim = featureDim

    channels = levels * featureDim
    self.layers = nn.Sequential(
      nn.Conv2d(channels, channels, kernel_size=1, groups=levels),
      nn.ReLU(),
      nn.Conv2d(channels, channels, kernel_size=1, groups=levels),
    )

  def forward(self, state):
    # drop the final level (bottom up doesn't do anything to the top level), and flatten level, feature dims
    x = state[:, :-1].flatten(1, 2)
    x = self.layers(x)
    # unflatten level & feature dims
    x = x.unflatten(1, (self.levels, self.featureDim))
    # add the final level back
    return torch.cat([x, torch.zeros_like(x[:, 0]).unsqueeze(1)], dim=1)

class TopDown(nn.Module):
  def __init__(self, levels, featureDim, positionalDim):
    super().__init__()
    self.levels = levels
    self.featureDim = featureDim

    channelsIn = levels * (featureDim + positionalDim)
    channelsOut = levels * featureDim
    self.layers = nn.Sequential(
      nn.Conv2d(channelsIn, channelsOut, kernel_size=1, groups=levels),
      nn.ReLU(),
      nn.Conv2d(channelsOut, channelsOut, kernel_size=1, groups=levels),
    )

  def forward(self, state, positional):
    # slice off input level (not modified), and flatten level & feature dims
    batch = state.shape[0]
    # expand the positional encoding for concat
    positional = positional.expand(batch, self.levels, -1, -1, -1)
    x = torch.cat([state[:, 1:], positional], dim=2).flatten(1, 2)
    x = self.layers(x)
    # unflatten level & feature dimensions
    x = x.unflatten(1, (self.levels, self.featureDim))
    # add zeros back for the input level
    return torch.cat([torch.zeros_like(x[:, 0]).unsqueeze(1), x], dim=1)

# naive pairwise attention
class Attention(nn.Module):
  def __init__(self, levels, featureDim):
    super().__init__()
    self.levels = levels
    self.featureDim = featureDim

  def forward(self, x, beta=1):
    # x has shape [batch, levels, featureDim, height, width]
    # -> [batch, levels, height, width, featureDim]
    x = x.permute(0, 1, 3, 4, 2)
    # -> [batch, levels, height * width, featureDim]
    x = x.flatten(2, 3)
    # batched pairwise dot-product; -> shape [batch, levels, height * width, height * width]
    product = torch.matmul(x, x.transpose(2, 3))
    expProduct = torch.exp(beta * product)
    # attention weights; -> shape [batch, levels, height * width, height * width]
    attention = (expProduct / expProduct.sum(dim=3, keepdim=True))
    return (x.unsqueeze(2) * attention.unsqueeze(4)).sum(dim=3)

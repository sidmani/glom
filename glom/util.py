import torch
import numpy as np

# positional encoding (as in NeRF, Transformer)
def positionalEncoding(resolution, count, device=None):
  # the basis: [pi, 2pi, 4pi, 8pi...]
  basis = (2.0 ** (torch.linspace(0, count - 1, count, device=device))) * np.pi
  basis = basis.view(1, 1, 1, -1)

  # create the grid
  edge = torch.linspace(0, 1, resolution, device=device)
  grid = torch.stack(torch.meshgrid(edge, edge), dim=2).unsqueeze(3)
  arg = (grid * basis).flatten(2, 3)
  sin, cos = torch.sin(arg), torch.cos(arg)
  return torch.cat([sin, cos], dim=2).permute(2, 0, 1)

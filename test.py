import glom.modules as M
import torch
from glom.util import positionalEncoding

feature_dim = 12
batch = 3
levels = 5
edge = 64

BU = M.BottomUp(levels, feature_dim)
inp = torch.zeros(batch, levels + 1, feature_dim, edge, edge)
ret = BU(inp)
print(ret.shape)

positional = positionalEncoding(edge, 7)
positionalDim = 7 * 2 * 2
TD = M.TopDown(levels, feature_dim, positionalDim)
ret2 = TD(inp, positional)
print(ret2.shape)

Attn = M.Attention(levels, feature_dim)
ret3 = Attn(inp)
print(ret3.shape)

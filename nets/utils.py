import torch.nn as nn
from torch.nn import Parameter
from torch import norm_except_dim

class WeightNorm(nn.Module):
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, name="weight", dim=0, eps=0.0):
        super(WeightNorm, self).__init__()
        self.module = module
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim
        self.eps = eps
        self.name_g = self.name + self.append_g
        self.name_v = self.name + self.append_v
        self._reset()

    def _reset(self):
        weight = getattr(self.module, self.name)

        # construct g,v such that w = g/||v|| * v
        g = Parameter(norm_except_dim(weight, 2, self.dim).data)
        v = Parameter(weight.data)

        # remove w from parameter list
        del self.module._parameters[self.name]

        # add g and v as new parameters
        self.module.register_parameter(self.name_g, g)
        self.module.register_parameter(self.name_v, v)

    def _setweights(self):
        g = getattr(self.module, self.name_g)
        v = getattr(self.module, self.name_v)
        w = v * g / (norm_except_dim(v, 2, self.dim) + self.eps)
        setattr(self.module, self.name, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

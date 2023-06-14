import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from scipy.optimize import bisect
from scipy.optimize import newton

class Flow1d(nn.Module):
    def __init__(self, n_components):
        super(Flow1d, self).__init__()
        self.mus = nn.Parameter(torch.randn(n_components), requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        self.weight_logits = nn.Parameter(torch.ones(n_components), requires_grad=True)

    def forward(self, x):
        x = x.view(-1,1)
        weights = self.weight_logits.softmax(dim=0).view(1,-1)
        distribution = Normal(self.mus, self.log_sigmas.exp())
        z = (distribution.cdf(x) * weights).sum(dim=1)
        log_dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(dim=1).log()
        return z, log_dz_by_dx

    def inverse(self, z):
        if isinstance(z, tuple):
            z, _ = z
        def f(x):
            return self.forward(torch.tensor(x).unsqueeze(0))[0] - z
        x = bisect(f,-20,20)
        return torch.tensor(x).reshape(z.shape), None

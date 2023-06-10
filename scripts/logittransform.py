import torch
import torch.nn as nn

class LogitTransform(nn.Module):
    def __init__(self, alpha):
        super(LogitTransform, self).__init__()
        self.alpha = alpha 

    def forward(self, x):
        x_new = self.alpha/2 + (1-self.alpha)*x 
        z = torch.log(x_new) - torch.log(1-x_new)
        log_dz_by_dx = torch.log(torch.FloatTensor([1-self.alpha])) - torch.log(x_new) - torch.log(1-x_new)
        return z, log_dz_by_dx

    def inverse(self, z):
        if not isinstance(z, tuple):
            z = (z, None)
        z, _ = z
        x_new = torch.sigmoid(z)
        return (x_new - self.alpha/2) / (1-self.alpha), None
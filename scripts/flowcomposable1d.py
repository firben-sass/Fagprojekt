import torch.nn as nn

class FlowComposable1d(nn.Module):
    def __init__(self, flow_models_list):
        super(FlowComposable1d, self).__init__()
        self.flow_models_list = nn.ModuleList(flow_models_list)

    def forward(self, x):
        z, sum_log_dz_by_dx = x, 0
        for flow in self.flow_models_list:
            z, log_dz_by_dx = flow(z)
            sum_log_dz_by_dx += log_dz_by_dx
        return z, sum_log_dz_by_dx

    def inverse(self, z):
        if not isinstance(z, tuple):
            z = (z, None)
        x, _ = z
        for flow in reversed(self.flow_models_list):
            x, _ = flow.inverse((x, _))
        return x, None
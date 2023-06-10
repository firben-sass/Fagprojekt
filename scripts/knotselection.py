import torch
import numpy as np

class KnotSelection:
    def __init__(self, num_knots, flow_model, data):
        self.num_knots = num_knots
        self.flow_model = flow_model
        self.data = data
    
    def inverseflow(self, knots_z):
        knots_x = []
        with torch.no_grad():
            for knot in knots_z:
                knot_tensor = torch.tensor(knot).unsqueeze(0)
                x = self.flow_model.inverse(knot_tensor)
                if isinstance(x, tuple):
                    knot_x = x[0].item()
                else:
                    knot_x = x.item()
                knots_x.append(knot_x)
        return knots_x

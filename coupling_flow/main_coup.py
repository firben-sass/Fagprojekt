#%%

import normflows as nf
from torch.optim import optimizergi

def coupling_aff(data):
    base = nf.distributions.base.Uniform(-1, 1)

    # Define list of flows
    num_layers = 32
    flows = []
    for i in range(num_layers):
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(2, mode='swap'))


    # If the target density is not given
    model = nf.NormalizingFlow(base, flows)
    
    # When doing maximum likelihood learning, i.e. minimizing the forward KLD
    # with no target distribution given
    loss = model.forward_kld(x)

    # When minimizing the reverse KLD based on the given target distribution
    loss = model.reverse_kld(num_samples=512)

    # Optimization as usual
    loss.backward()
    optimizer.step()
    
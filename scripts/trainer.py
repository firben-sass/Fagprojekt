from scripts.flow1d import Flow1d
from scripts.logittransform import LogitTransform
from scripts.flowcomposable1d import FlowComposable1d
from tqdm import tqdm
import torch

class Trainer:
    def __init__(self, target_distribution, train_loader, test_loader):
        self.target_distribution = target_distribution
        self.train_loader = train_loader
        self.test_loader = test_loader

    def loss_function(self, z, log_dz_dx):
        log_likelihood = self.target_distribution.log_prob(z) + log_dz_dx
        negative_log_likelihood = -log_likelihood.mean()
        return negative_log_likelihood

    def train(self, model, optimizer):
        model.train()
        for x in self.train_loader:
            z, log_dz_dx = model(x)
            loss = self.loss_function(z, log_dz_dx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def eval_loss(self, model, data_loader):
        model.eval()
        total_loss = 0
        for x in data_loader:
            z, log_dz_dx = model(x)
            loss = self.loss_function(z, log_dz_dx)
            total_loss += loss * x.size(0)
        return (total_loss / len(data_loader.dataset)).item()

    def flow_model(self, layer_sizes):
        flow_models_list = []
        alpha = 0.1
        for i in range(len(layer_sizes)-1):
            flow_models_list.append(Flow1d(layer_sizes[i]))
            flow_models_list.append(LogitTransform(alpha))
        flow_models_list.append(Flow1d(layer_sizes[-1]))
        return flow_models_list

    def train_and_eval(self, epochs, lr):
        layer_sizes = [64, 64, 64]
        flow_model = self.flow_model(layer_sizes)
        flow = FlowComposable1d(flow_model)
        optimizer = torch.optim.Adam(flow.parameters(), lr = lr)
        train_losses = []
        test_losses = []
        for it in tqdm(range(epochs)):
            self.train(flow, optimizer=optimizer)
            train_loss = self.eval_loss(flow, self.train_loader)
            test_loss = self.eval_loss(flow, self.test_loader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
        return flow, train_losses, test_losses

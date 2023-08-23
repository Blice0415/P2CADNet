import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import utils

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 200)
        self.fc2_mu = nn.Linear(200, 10)
        self.fc2_log_std = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 200)
        self.fc4 = nn.Linear(200, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        recon = torch.sigmoid(self.fc4(h3))  # use sigmoid because the input image's pixel is between 0-1
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_std = self.encode(x)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z)
        return recon, mu, log_std

    def loss_function(self, recon, x, mu, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")  # use "mean" may have a bad effect on gradients
        kl_loss = -0.5 * (1 + 2*log_std - mu.pow(2) - torch.exp(2*log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss

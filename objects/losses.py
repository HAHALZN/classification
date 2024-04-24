import torch
import torch.nn as nn
import torch.nn.functional as F


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1).mean()
        return b


class EntropyDiv(nn.Module):
    def __init__(self):
        super(EntropyDiv, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1).mean(dim=0)
        b = torch.sum(b * torch.log(b + 1e-5))
        return b

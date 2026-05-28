import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv 

class GNNModel(nn.Module):
    def __init__(self, input_dim=7):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return torch.sigmoid(x)
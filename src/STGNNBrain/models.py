import torch
from torch_geometric.nn import GCNConv, ChebConv, GATConv, SGConv
import numpy as np
import os

import torch.nn as nn
import torch.nn.functional as F

class STGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(STGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x, _ = self.lstm(x.unsqueeze(0))
        return self.linear(x.squeeze(0))

class STChebNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=3):
        super(STChebNet, self).__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x, _ = self.gru(x.unsqueeze(0))
        return self.linear(x.squeeze(0))

class STGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(STGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads)
        self.temporal_conv = nn.Conv1d(hidden_channels * heads, hidden_channels, kernel_size=3, padding=1)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.temporal_conv(x.unsqueeze(0).transpose(1, 2)))
        return self.linear(x.squeeze(0).transpose(1, 2))

class STSGConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=3):
        super(STSGConv, self).__init__()
        self.conv1 = SGConv(in_channels, hidden_channels, K)
        self.conv2 = SGConv(hidden_channels, hidden_channels, K)
        self.temporal_attn = nn.MultiheadAttention(hidden_channels, num_heads=4)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = x.unsqueeze(0)
        x, _ = self.temporal_attn(x, x, x)
        return self.linear(x.squeeze(0))
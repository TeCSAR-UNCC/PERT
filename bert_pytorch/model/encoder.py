import torch
import math
import numpy as np
from tqdm import tqdm
from torch import nn

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from dataset.graph import JOINTS_DEF, JOINTS_PAIRS

class GraphConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(GraphConvBlock, self).__init__()
        self.conv = GCNConv(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GCNEncoder(nn.Module):
    def __init__(self, arch, token_window_size,
                 out_features, in_kp=15, dropout=0.3):
        super(GCNEncoder, self).__init__()

        edge_index = self.create_edge_tensor(token_window_size, True)
        self.register_buffer('edge_index', edge_index)

        self.fc_output_dim = out_features
        self.token_window_size = token_window_size
        self.layers = nn.ModuleList()
        for input_dim, output_dim in arch:
            self.layers.append(GraphConvBlock(input_dim, output_dim, dropout))

        end_channels = arch[-1][-1]
        in_features = token_window_size * in_kp * end_channels
        self.fc = nn.Sequential(
            # nn.LazyLinear(1028),
            nn.Linear(in_features, out_features)
        )
        

    def forward(self, x):
        # Input shape (Batch, Window, Keypoints, xy)
        batch, window, keypoints, channels = x.shape
        tokens = int(window / self.token_window_size)
        x = x.view(batch, tokens, self.token_window_size, keypoints, channels)
        x = x.view(batch * tokens, self.token_window_size * keypoints, channels)

        for layer in self.layers:
            x = layer(x, self.edge_index)

        x = x.reshape(batch, tokens, -1)
        x = self.fc(F.relu(x))

        # Output shape (Batch, Tokens, fc_output_dim)
        return x
    
    def create_edge_tensor(self, num_people, temporal_edges=False):
        """
        Creates an edge tensor for a graph with varying numbers of people.

        Args:
            num_people (int): Number of people (skeletons) in the window.
            temporal_edges (bool): If True, add temporal edges between consecutive time steps.

        Returns:
            torch.Tensor: Edge tensor representing the graph connections.
        """

        edge_list = []
        keypoints = len(JOINTS_DEF)

        # Adding spatial edges for each person's skeleton
        for person in range(num_people):
            offset = person * keypoints
            for start_joint, end_joint in JOINTS_PAIRS:
                start_idx = JOINTS_DEF[start_joint] + offset
                end_idx = JOINTS_DEF[end_joint] + offset
                edge_list.append((start_idx, end_idx))
                edge_list.append((end_idx, start_idx))  # Adding reverse connection

        # Adding temporal edges if required
        if temporal_edges:
            for person in range(num_people - 1):  # Iterate until the second last person
                offset_current = person * keypoints
                offset_next = (person + 1) * keypoints
                for kp in range(keypoints):
                    current_time_kp = kp + offset_current
                    next_time_kp = kp + offset_next
                    edge_list.append((current_time_kp, next_time_kp))
                    edge_list.append((next_time_kp, current_time_kp))  # Adding reverse connection

        # Convert edge list to a tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        return edge_index

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
    def __init__(self, arch, token_window_size, out_features, in_kp=15, dropout=0.3):
        super(GCNEncoder, self).__init__()

        # Create edge tensor for graph convolution and register it as a buffer
        edge_index = self.create_edge_tensor(token_window_size, True)
        self.register_buffer('edge_index', edge_index)

        # Store output dimension for later use in the forward pass
        self.fc_output_dim = out_features

        # Store token window size for reshaping input in forward pass
        self.token_window_size = token_window_size

        # Create a list of Graph Convolutional Layers based on the provided architecture
        self.layers = nn.ModuleList()
        for input_dim, output_dim in arch:
            self.layers.append(GraphConvBlock(input_dim, output_dim, dropout))

        # Calculate the number of features from the last layer of GCN
        end_channels = arch[-1][-1]

        # Calculate the input features for the fully connected layer
        in_features = token_window_size * in_kp * end_channels

        # Define the final fully connected layer for encoding
        self.fc = nn.Sequential(
            # nn.LazyLinear(1028),  # Uncomment if you want to use LazyLinear
            nn.Linear(in_features, out_features)
        )

    def forward(self, x):
        # x, org = self._tokenize(x)

        # batch, tokens, *_ = org
        batch, tokens, keypoints, channels = x.shape
        x = x.view(batch * tokens, keypoints, channels)

        # Pass input through each Graph Convolutional Layer
        for layer in self.layers:
            x = layer(x, self.edge_index)

        # Reshape the output and apply ReLU activation
        x = x.reshape(batch, tokens, -1)
        x = self.fc(F.relu(x))

        # Output shape (Batch, Tokens, fc_output_dim)
        return x

    def _tokenize(self, x):
        # Input shape (Batch, Window, Keypoints, Channels)
        batch, window, keypoints, channels = x.shape

        # Calculate the number of tokens in each window
        tokens = int(window / self.token_window_size)

        # Reshape input to process in tokens
        x = x.view(batch, tokens, self.token_window_size, keypoints, channels)
        org = (batch, tokens, self.token_window_size, keypoints, channels)
        x = x.view(batch * tokens, self.token_window_size * keypoints, channels)

        return x, org
    
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

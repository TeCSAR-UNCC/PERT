import torch
import torch.nn as nn
import os

from bert_pytorch.model.transformer import TransformerBlock
from bert_pytorch.model.embedding.position import PositionalEmbedding


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.train_epoch_loss = []
        self.test_epoch_loss = []

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        self.position = PositionalEmbedding(d_model=self.hidden)

    def forward(self, x, padding=None):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, 1, seq_len])
        
        # Create a mask with 'True' values at positions we want to mask
        mask = torch.arange(x.size(1), device=x.device)
        mask = mask.expand(x.size(0), x.size(1)) 
        mask = mask >= padding.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(2)

        # Positional Encoding
        x = self.position(x)
        
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
    
    
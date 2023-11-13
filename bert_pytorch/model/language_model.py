import torch
import torch.nn as nn

from .bert import BERT


class PERT_Mask_Order(nn.Module):
    """
    PERT Pose Mask
    Masked Tokens + Token Order 
    """

    def __init__(self, pert: BERT, token_window_size, add_cls=True, num_cls=2):
        """
        :param pert: BERT model which should be trained
        :param num_cls: total vocab size for masked_lm
        """

        super().__init__()
        self.pert = pert
        self.add_cls = add_cls
        self.num_cls = num_cls
        self.mask_chance = 0.2
        self.mix_chance = 0.5
        self.token_window_size = token_window_size
        self.classification = ClassificationModel(self.pert.regout, self.num_cls)
        self.softmax = torch.nn.Softmax(dim=1)
        # self.mask_lm = MaskedPoseModel(self.pert.hidden)

    def forward(self, x, mask_tokens, padding=None):

        x = self.pert.emb(x)
        x = self.pert.pos(x)  # inject position info

        batch_size, seq_len, *_ = x.size()
        
        mask = None
        mask = torch.arange(x.size(1), device=x.device)
        mask = mask.expand(x.size(0), x.size(1)) 
        mask = mask >= padding.unsqueeze(1)

        # Transformer
        init = [self.pert.init_pose.repeat(batch_size, seq_len, 1)]  # init mean pose
        y = self.pert.decoder(x, init, mask)

        mixed_output = self.classification(y[:, 0])
        mixed_output = self.softmax(mixed_output)

        masked_output = y[mask_tokens].view(batch_size, -1, self.pert.out_kp, self.pert.out_chns)
        
        return masked_output, mixed_output
    

class ClassificationModel(nn.Module):
    def __init__(self, hidden, num_cls):
        super().__init__()
        self.linear = nn.Linear(hidden, num_cls)

    def forward(self, x):
        return self.linear(x)

class MaskedPoseModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.linear(x)

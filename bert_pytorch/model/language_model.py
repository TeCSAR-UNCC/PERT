import torch.nn as nn

from .bert import BERT


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, num_cls=120):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.num_cls = num_cls
        self.classification = ClassificationModel(self.bert.hidden, self.num_cls)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden)

    def forward(self, x, padding=None):
        x = self.bert(x, padding)
        return self.mask_lm(x), self.classification(x[:, 0])


class ClassificationModel(nn.Module):
    def __init__(self, hidden, num_cls):
        super().__init__()
        self.linear = nn.Linear(hidden, num_cls)

    def forward(self, x):
        return self.linear(x)

class MaskedLanguageModel(nn.Module):
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

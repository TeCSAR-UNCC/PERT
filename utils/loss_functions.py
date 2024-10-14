import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.spatial.distance import cosine


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def spatial_cosine_similarity(pred, target):
    """
    Calculate the cosine similarity for spatial analysis with an additional tokens dimension.
    Input tensors are of shape (batch, tokens, window, keypoints, xy).
    Returns a tensor of cosine similarities of shape (batch, tokens, window).
    """
    # Reshape to (batch*tokens*window, keypoints*xy)
    pred_flat = pred.view(pred.shape[0] * pred.shape[1] * pred.shape[2], -1)
    target_flat = target.view(target.shape[0] * target.shape[1] * target.shape[2], -1)

    # Normalize the vectors
    pred_norm = F.normalize(pred_flat, p=2, dim=1)
    target_norm = F.normalize(target_flat, p=2, dim=1)

    # Calculate cosine similarity
    cosine_sim = torch.mm(pred_norm, target_norm.t())

    # Reshape to get similarity for each window in each token in each batch
    cosine_loss = (
        cosine_sim.diag()
        .view(pred.shape[0], pred.shape[1], pred.shape[2])
        .mean(dim=[2])
    )
    one_tensor = torch.ones_like(cosine_loss, device=cosine_loss.device)
    return one_tensor - cosine_loss


def cosine_similarity_loss(batch1, batch2):
    """
    Calculate cosine similarity between datapoints in two batches.

    Args:
    - batch1: Tensor of shape (batch_size, feature_dim) for the first batch.
    - batch2: Tensor of shape (batch_size, feature_dim) for the second batch.

    Returns:
    - similarities: Tensor of shape (batch_size,) containing cosine similarities for each pair of datapoints.
    """
    batch_size, *_ = batch1.size()
    batch1 = batch1.reshape(batch_size, -1)
    batch2 = batch2.reshape(batch_size, -1)

    # Normalize the input tensors
    batch1 = F.normalize(batch1, p=2, dim=1)
    batch2 = F.normalize(batch2, p=2, dim=1)

    # Calculate the dot product
    dot_product = torch.sum(batch1 * batch2, dim=1)
    one_tensor = torch.ones(dot_product.shape, device=dot_product.device)
    # Calculate cosine similarity
    similarities = dot_product

    return one_tensor - similarities


def temporal_mse_loss(pred, target):
    """
    Calculate the temporal mean squared error loss.
    Input tensors are of shape (batch, time, keypoints, xy).
    Returns a tensor of MSE loss of shape (batch).
    """
    # Calculate movement vectors (deltas) for pred and target
    pred_deltas = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    target_deltas = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]

    # Calculate MSE loss
    loss = nn.MSELoss(reduction="none")(pred_deltas, target_deltas)
    return loss.mean(dim=[2, 3, 4])


def combined_evaluation(pred, target):
    """
    Combine the results of spatial cosine similarity and temporal MSE loss.
    Input tensors are of shape (batch, time, keypoints, xy).
    Returns a tensor of combined results of shape (batch).
    """
    cosine_similarity = spatial_cosine_similarity(pred, target)
    temporal_mse = temporal_mse_loss(pred, target)
    spacial_loss = nn.MSELoss(reduction="none")(pred, target).mean(dim=[2, 3, 4])

    # Combine by adding (or any other combination logic)
    combined_result = cosine_similarity + 10 * temporal_mse + spacial_loss

    return combined_result


# Assume pred and target are PyTorch tensors of the shape (batch, time, keypoints, xy)
# pred = torch.ones(2, 9, 15, 15, 2) * 0.2  # Example predicted data
# target = torch.ones(2, 9, 15, 15, 2) * 0.1  # Example ground truth data

# combined_results = combined_evaluation(pred, target)
# print("Combined Results:", combined_results)

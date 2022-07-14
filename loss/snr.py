import torch
import torch.nn as nn
import torch.nn.functional as F

import sampling

ALLOWED_MINING_OPS = list(sampling.BATCHMINING_METHODS.keys())
REQUIRES_SAMPLING = True
REQUIRES_OPTIM = False


class Criterion(nn.Module):
    """
    This implements the Signal-To-Noise Ratio Triplet Loss
    """

    def __init__(self, opt, sampling):
        super(Criterion, self).__init__()
        self.margin = opt.loss_snr_margin
        self.reg_lambda = opt.loss_snr_reg_lambda
        self.sampling = sampling

        if self.sampling.name == 'distance': self.reg_lambda = 0

        self.name = 'snr'

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_SAMPLING = REQUIRES_SAMPLING
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def forward(self, batch, labels, **kwargs):
        sampled_triplets = self.sampling(batch, labels)
        anchors = [triplet[0] for triplet in sampled_triplets]
        positives = [triplet[1] for triplet in sampled_triplets]
        negatives = [triplet[2] for triplet in sampled_triplets]

        pos_snr = torch.var(batch[anchors, :] - batch[positives, :], dim=1) / torch.var(batch[anchors, :], dim=1)
        neg_snr = torch.var(batch[anchors, :] - batch[negatives, :], dim=1) / torch.var(batch[anchors, :], dim=1)

        reg_loss = torch.mean(torch.abs(torch.sum(batch[anchors, :], dim=1)))

        snr_loss = F.relu(pos_snr - neg_snr + self.margin)
        snr_loss = torch.sum(snr_loss) / torch.sum(snr_loss > 0)

        loss = snr_loss + self.reg_lambda * reg_loss

        return loss

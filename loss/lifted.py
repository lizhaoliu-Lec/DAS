import torch
import torch.nn as nn
import torch.nn.functional as F

ALLOWED_MINING_OPS = ['lifted']
REQUIRES_SAMPLING = True
REQUIRES_OPTIM = False


class Criterion(nn.Module):
    def __init__(self, opt, sampling):
        super(Criterion, self).__init__()
        self.margin = opt.loss_lifted_neg_margin
        self.l2_weight = opt.loss_lifted_l2
        self.sampling = sampling

        self.name = 'lifted'

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_SAMPLING = REQUIRES_SAMPLING
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def forward(self, batch, labels, **kwargs):
        anchors, positives, negatives = self.sampling(batch, labels)

        loss = []
        for anchor, positive_set, negative_set in zip(anchors, positives, negatives):
            anchor, positive_set, negative_set = batch[anchor, :].view(1, -1), batch[positive_set, :].view(1, len(
                positive_set), -1), batch[negative_set, :].view(1, len(negative_set), -1)
            pos_term = torch.logsumexp(nn.PairwiseDistance(p=2)(anchor[:, :, None], positive_set.permute(0, 2, 1)),
                                       dim=1)
            neg_term = torch.logsumexp(
                self.margin - nn.PairwiseDistance(p=2)(anchor[:, :, None], negative_set.permute(0, 2, 1)), dim=1)
            loss.append(F.relu(pos_term + neg_term))

        loss = torch.mean(torch.stack(loss)) + self.l2_weight * torch.mean(torch.norm(batch, p=2, dim=1))
        return loss

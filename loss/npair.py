import torch
import torch.nn as nn

ALLOWED_MINING_OPS = ['npair']
REQUIRES_SAMPLING = True
REQUIRES_OPTIM = False


class Criterion(nn.Module):
    def __init__(self, opt, sampling):
        """
        Args:
        """
        super(Criterion, self).__init__()
        self.pars = opt
        self.l2_weight = opt.loss_npair_l2
        self.sampling = sampling

        self.name = 'npair'

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_SAMPLING = REQUIRES_SAMPLING
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def forward(self, batch, labels, **kwargs):
        anchors, positives, negatives = self.sampling(batch, labels)

        loss = 0
        if 'bninception' in self.pars.arch:
            # clamping/value reduction to avoid initial overflow for high embedding dimensions!
            batch = batch / 4
        for anchor, positive, negative_set in zip(anchors, positives, negatives):
            a_embs, p_embs, n_embs = batch[anchor:anchor + 1], batch[positive:positive + 1], batch[negative_set]
            inner_sum = a_embs[:, None, :].bmm((n_embs - p_embs[:, None, :]).permute(0, 2, 1))
            inner_sum = inner_sum.view(inner_sum.shape[0], inner_sum.shape[-1])
            loss = loss + torch.mean(torch.log(torch.sum(torch.exp(inner_sum), dim=1) + 1)) / len(anchors)
            loss = loss + self.l2_weight * torch.mean(torch.norm(batch, p=2, dim=1)) / len(anchors)

        return loss

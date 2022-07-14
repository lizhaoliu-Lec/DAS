import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss._share import bl_proxy_weight

ALLOWED_MINING_OPS = None
REQUIRES_SAMPLING = False
REQUIRES_OPTIM = True


class Criterion(nn.Module):
    """
    This implementation follows the pseudocode provided in the original paper.
    """

    def __init__(self, opt):
        super(Criterion, self).__init__()
        self.par = opt
        self.opt = opt

        self.angular_margin = opt.loss_arcface_angular_margin
        self.feature_scale = opt.loss_arcface_feature_scale

        self.class_map = nn.Parameter(torch.Tensor(opt.n_classes, opt.embed_dim))
        stdv = 1. / np.sqrt(self.class_map.size(1))
        self.class_map.data.uniform_(-stdv, stdv)

        self.name = 'arcface'

        self.lr = opt.loss_arcface_lr

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_SAMPLING = REQUIRES_SAMPLING
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def forward(self, batch, labels, **kwargs):
        labels = labels.to(self.par.device)

        bs = len(batch)

        class_map = F.normalize(self.class_map, dim=1)
        # Note that the similarity becomes the cosine for normalized embeddings.
        # Denoted as 'fc7' in the paper pseudocode.
        cos_similarity = batch.mm(class_map.T).clamp(min=1e-10, max=1 - 1e-10)

        pick = torch.zeros(bs, self.par.n_classes).byte().to(self.par.device)
        pick[torch.arange(bs), labels] = 1

        original_target_logit = cos_similarity[pick]

        theta = torch.acos(original_target_logit)
        marginal_target_logit = torch.cos(theta + self.angular_margin)

        class_pred = self.feature_scale * (
                cos_similarity + (marginal_target_logit - original_target_logit).unsqueeze(1))
        loss = nn.CrossEntropyLoss()(class_pred, labels)

        return loss

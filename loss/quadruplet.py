import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sampling

ALLOWED_MINING_OPS = list(sampling.BATCHMINING_METHODS.keys())
REQUIRES_SAMPLING = True
REQUIRES_OPTIM = False


class Criterion(nn.Module):
    def __init__(self, opt, sampling):
        super(Criterion, self).__init__()
        self.sampling = sampling

        self.name = 'quadruplet'

        self.margin_alpha_1 = opt.loss_quadruplet_margin_alpha_1
        self.margin_alpha_2 = opt.loss_quadruplet_margin_alpha_2

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_SAMPLING = REQUIRES_SAMPLING
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def triplet_distance(self, anchor, positive, negative):
        return F.relu(torch.norm(anchor - positive, p=2, dim=-1) - torch.norm(anchor - negative, p=2,
                                                                              dim=-1) + self.margin_alpha_1)

    def quadruplet_distance(self, anchor, positive, negative, fourth_negative):
        return F.relu(
            torch.norm(anchor - positive, p=2, dim=-1) - torch.norm(negative - fourth_negative, p=2,
                                                                    dim=-1) + self.margin_alpha_2)

    def forward(self, batch, labels, **kwargs):
        sampled_triplets = self.sampling(batch, labels)

        anchors = np.array([triplet[0] for triplet in sampled_triplets]).reshape(-1, 1)
        positives = np.array([triplet[1] for triplet in sampled_triplets]).reshape(-1, 1)
        negatives = np.array([triplet[2] for triplet in sampled_triplets]).reshape(-1, 1)

        fourth_negatives = negatives != negatives.T
        fourth_negatives = [np.random.choice(np.arange(len(batch))[idxs]) for idxs in fourth_negatives]

        triplet_loss = self.triplet_distance(batch[anchors, :], batch[positives, :], batch[negatives, :])
        quadruplet_loss = self.quadruplet_distance(batch[anchors, :], batch[positives, :], batch[negatives, :],
                                                   batch[fourth_negatives, :])

        return torch.mean(triplet_loss) + torch.mean(quadruplet_loss)

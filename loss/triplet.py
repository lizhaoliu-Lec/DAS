import torch
import torch.nn as nn
import torch.nn.functional as F

import sampling
from loss._share import bl_triplets_weight

ALLOWED_MINING_OPS = list(sampling.BATCHMINING_METHODS.keys())
REQUIRES_SAMPLING = True
REQUIRES_OPTIM = False


class Criterion(nn.Module):
    """
    Standard Triplet Loss, finds triplets in Mini-batches.
    """

    def __init__(self, opt, sampling):
        super(Criterion, self).__init__()
        self.opt = opt
        self.margin = opt.loss_triplet_margin
        self.sampling = sampling
        self.name = 'triplet'

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_SAMPLING = REQUIRES_SAMPLING
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def triplet_distance(self, anchor, positive, negative):
        return F.relu(
            (anchor - positive).pow(2).sum() - (anchor - negative).pow(2).sum() + self.margin)

    def forward(self, batch, labels, **kwargs):
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        sampled_triplets = self.sampling(batch, labels)

        # if tensorboard in kwargs
        tensorboard = kwargs.get('tensorboard', None)
        if tensorboard and batch.size(0) > self.opt.bs:
            if self.opt.embedding_producer_used:
                bs = self.opt.bs
                all_idx = [t for triplet in sampled_triplets for t in triplet]
                num_generated = sum([1 if _ >= bs else 0 for _ in all_idx])
                num_real = sum([0 if _ >= bs else 1 for _ in all_idx])
                tensorboard.add_scalar(tag='EmbeddingProducer/NumReal', scalar_value=num_real,
                                       global_step=self.opt.iteration)
                tensorboard.add_scalar(tag='EmbeddingProducer/NumGenerated', scalar_value=num_generated,
                                       global_step=self.opt.iteration)
                tensorboard.add_scalar(tag='EmbeddingProducer/RealVsGenerated', scalar_value=num_real / num_generated,
                                       global_step=self.opt.iteration)

        loss = torch.stack([
            self.triplet_distance(batch[triplet[0]], batch[triplet[1]], batch[triplet[2]])
            for triplet in sampled_triplets
        ])

        loss = torch.mean(loss)

        return loss

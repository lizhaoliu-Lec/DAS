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
        self.pos_margin = opt.loss_contrastive_pos_margin
        self.neg_margin = opt.loss_contrastive_neg_margin
        self.stop_pos = opt.loss_contrastive_stop_pos
        self.stop_neg = opt.loss_contrastive_stop_neg
        self.sampling = sampling
        self.opt = opt

        self.name = 'contrastive'

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_SAMPLING = REQUIRES_SAMPLING
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def forward(self, batch, labels, **kwargs):
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

        anchors = [triplet[0] for triplet in sampled_triplets]
        positives = [triplet[1] for triplet in sampled_triplets]
        negatives = [triplet[2] for triplet in sampled_triplets]

        if not self.stop_pos:
            pos_dists = torch.mean(
                F.relu(nn.PairwiseDistance(p=2)(batch[anchors, :], batch[positives, :]) - self.pos_margin))
        else:
            pos_dists = torch.mean(
                F.relu(nn.PairwiseDistance(p=2)(batch[anchors, :], batch[positives, :].detach()) - self.pos_margin))
        if not self.stop_neg:
            neg_dists = torch.mean(
                F.relu(self.neg_margin - nn.PairwiseDistance(p=2)(batch[anchors, :], batch[negatives, :])))
        else:
            neg_dists = torch.mean(
                F.relu(self.neg_margin - nn.PairwiseDistance(p=2)(batch[anchors, :], batch[negatives, :].detach())))

        loss = pos_dists + neg_dists

        return loss

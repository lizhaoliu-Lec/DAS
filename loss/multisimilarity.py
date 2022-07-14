import torch
import torch.nn as nn

ALLOWED_MINING_OPS = None
REQUIRES_SAMPLING = False
REQUIRES_OPTIM = False


class Criterion(nn.Module):
    def __init__(self, opt):
        super(Criterion, self).__init__()
        self.n_classes = opt.n_classes
        self.opt = opt
        self.pos_weight = opt.loss_multisimilarity_pos_weight
        self.neg_weight = opt.loss_multisimilarity_neg_weight
        self.margin = opt.loss_multisimilarity_margin
        self.thresh = opt.loss_multisimilarity_thresh

        self.name = 'multisimilarity'

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_SAMPLING = REQUIRES_SAMPLING
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def forward(self, batch, labels, **kwargs):
        similarity = batch.mm(batch.T)

        loss = []
        for i in range(len(batch)):

            pos_idxs = labels == labels[i]
            pos_idxs[i] = 0
            neg_idxs = labels != labels[i]

            # TODO figure out why this part cause the following error
            # RuntimeError: copy if failed to synchronize: an illegal memory access was encountered
            anchor_pos_sim = similarity[i][pos_idxs]
            anchor_neg_sim = similarity[i][neg_idxs]

            # This part doesn't really work, especially when you dont have a lot of positives in the batch...
            neg_idxs = (anchor_neg_sim + self.margin) > torch.min(anchor_pos_sim)
            pos_idxs = (anchor_pos_sim - self.margin) < torch.max(anchor_neg_sim)
            if not torch.sum(neg_idxs) or not torch.sum(pos_idxs):
                continue

            # if tensorboard in kwargs
            tensorboard = kwargs.get('tensorboard', None)
            if tensorboard and batch.size(0) > self.opt.bs:
                if self.opt.embedding_producer_used:
                    bs = self.opt.bs
                    num_real = (torch.sum(neg_idxs[:bs]) + torch.sum(pos_idxs[:bs])).item()
                    num_generated = (torch.sum(neg_idxs[bs:]) + torch.sum(pos_idxs[bs:])).item()
                    tensorboard.add_scalar(tag='EmbeddingProducer/NumReal', scalar_value=num_real,
                                           global_step=self.opt.iteration)
                    tensorboard.add_scalar(tag='EmbeddingProducer/NumGenerated', scalar_value=num_generated,
                                           global_step=self.opt.iteration)
                    RealVsGenerated = num_real / num_generated if num_generated > 0 else 0
                    tensorboard.add_scalar(tag='EmbeddingProducer/RealVsGenerated', scalar_value=RealVsGenerated,
                                           global_step=self.opt.iteration)

            anchor_neg_sim = anchor_neg_sim[neg_idxs]
            anchor_pos_sim = anchor_pos_sim[pos_idxs]

            pos_term = 1. / self.pos_weight * torch.log(
                1 + torch.sum(torch.exp(-self.pos_weight * (anchor_pos_sim - self.thresh))))
            neg_term = 1. / self.neg_weight * torch.log(
                1 + torch.sum(torch.exp(self.neg_weight * (anchor_neg_sim - self.thresh))))

            loss.append(pos_term + neg_term)

        loss = torch.mean(torch.stack(loss))
        return loss

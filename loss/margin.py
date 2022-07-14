import torch
import torch.nn as nn
import torch.nn.functional as F
import sampling
from loss._share import bl_triplets_weight

ALLOWED_MINING_OPS = list(sampling.BATCHMINING_METHODS.keys())
REQUIRES_SAMPLING = True
REQUIRES_OPTIM = True


class Criterion(nn.Module):
    """
    MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
    """

    def __init__(self, opt, sampling):
        super(Criterion, self).__init__()
        self.opt = opt
        self.n_classes = opt.n_classes

        self.margin = opt.loss_margin_margin
        self.nu = opt.loss_margin_nu
        self.beta_constant = opt.loss_margin_beta_constant
        self.beta_val = opt.loss_margin_beta

        if opt.loss_margin_beta_constant:
            self.beta = opt.loss_margin_beta
        else:
            self.beta = nn.Parameter(torch.ones(opt.n_classes) * opt.loss_margin_beta)

        self.sampling = sampling

        self.name = 'margin'

        self.lr = opt.loss_margin_beta_lr

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_SAMPLING = REQUIRES_SAMPLING
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def calculate_margin_distance_loss(self, anchor_batch, positive_batch, negative_batch, labels, sampled_triplets):
        if len(sampled_triplets):
            d_ap, d_an = [], []
            for triplet in sampled_triplets:
                train_triplet = {
                    'Anchor': anchor_batch[triplet[0]],
                    'Positive': positive_batch[triplet[1]],
                    'Negative': negative_batch[triplet[2]]
                }

                pos_dist = ((train_triplet['Anchor'] - train_triplet['Positive']).pow(2).sum() + 1e-8).pow(1 / 2)
                neg_dist = ((train_triplet['Anchor'] - train_triplet['Negative']).pow(2).sum() + 1e-8).pow(1 / 2)

                d_ap.append(pos_dist)
                d_an.append(neg_dist)

            d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

            if self.beta_constant:
                beta = self.beta
            else:
                beta = torch.stack(
                    [self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).to(torch.float).to(d_ap.device)

            pos_loss = F.relu(d_ap - beta + self.margin)
            neg_loss = F.relu(beta - d_an + self.margin)

            pair_count = torch.sum((pos_loss > 0.) + (neg_loss > 0.)).to(torch.float).to(d_ap.device)
            total_loss = pos_loss + neg_loss
            if pair_count == 0.:
                loss = torch.sum(total_loss)
            else:
                loss = torch.sum(total_loss) / pair_count

            if self.nu:
                beta_regularization_loss = torch.sum(beta)
                loss += self.nu * beta_regularization_loss.to(torch.float).to(d_ap.device)
        else:
            loss = torch.tensor(0.).to(torch.float).to(anchor_batch.device)

        return loss

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

        # vis the partition of the sampled real or generated samples

        loss = self.calculate_margin_distance_loss(batch, batch, batch, labels, sampled_triplets)

        return loss

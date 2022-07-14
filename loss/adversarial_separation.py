import torch
import torch.nn as nn
import torch.nn.functional as F

import sampling

ALLOWED_MINING_OPS = list(sampling.BATCHMINING_METHODS.keys())
REQUIRES_SAMPLING = False
REQUIRES_OPTIM = True


class Criterion(nn.Module):
    def __init__(self, opt):
        """
        MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super().__init__()

        self.embed_dim = opt.embed_dim
        self.proj_dim = opt.diva_decorrnet_dim

        self.directions = opt.diva_decorrelations
        self.weights = opt.diva_rho_decorrelation

        self.name = 'adversarial_separation'

        # Projection network
        self.regressors = nn.ModuleDict()
        for direction in self.directions:
            self.regressors[direction] = nn.Sequential(
                nn.Linear(self.embed_dim, self.proj_dim),
                nn.ReLU(),
                nn.Linear(self.proj_dim, self.embed_dim)
            ).to(torch.float).to(opt.device)

        # Learning Rate for Projection Network
        self.lr = opt.diva_decorrnet_lr

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_SAMPLING = REQUIRES_SAMPLING
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def forward(self, feature_dict):
        # Apply gradient reversal on input embeddings.
        adj_feature_dict = {key: F.normalize(grad_reverse(features), dim=-1) for key, features in
                            feature_dict.items()}
        # Project one embedding to the space of the other (with normalization), then compute the correlation.
        sim_loss = 0
        for weight, direction in zip(self.weights, self.directions):
            source, target = direction.split('-')
            sim_loss += -1. * weight * torch.mean(torch.mean((adj_feature_dict[target] * F.normalize(
                self.regressors[direction](adj_feature_dict[source]), dim=-1)) ** 2, dim=-1))
        return sim_loss


class GradRev(torch.autograd.Function):
    """
    Gradient Reversal Layer
    Implements an autograd class to flip gradients during backward pass.
    """

    def forward(self, x):
        """
        Container which applies a simple identity function.

        Input:
            x: any torch tensor input.
        """
        return x.view_as(x)

    def backward(self, grad_output):
        """
        Container to reverse gradient signal during backward pass.

        Input:
            grad_output: any computed gradient.
        """
        return grad_output * -1.


def grad_reverse(x):
    """
    Gradient reverse function
    Applies gradient reversal on input.

    Input:
        x: any torch tensor input.
    """
    return GradRev()(x)

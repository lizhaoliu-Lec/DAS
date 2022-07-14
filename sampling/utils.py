import numpy as np
import torch


def check_if_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def pair_wise_distance(A, clamp_min=0):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=clamp_min)
    return res.sqrt()


def inverse_sphere_distances(n, anchor_to_all_dists, labels, anchor_label):
    """
    Compute the inverse sphere distance.
    The Sphere Distance is: d^{n - 2} * (1 - 0.25 * d^2)^((n - 3) / 2).
    Reference:
    [1] Sampling Matters in Deep Embedding Learning (ICCV 2017). https://arxiv.org/abs/1706.07567
    [2] The sphere game in n dimensions.
    Then the inverse Sphere Distance is: d^{2 - n} * (1 - 0.25 * d^2)^((3 - n) / 2).
    """
    d = anchor_to_all_dists

    # negated log-distribution of distances of unit sphere in dimension <n>
    log_q_d_inv = ((2.0 - float(n)) * torch.log(d) - (float(n - 3) / 2) *
                   torch.log(1.0 - 0.25 * (d.pow(2))))
    log_q_d_inv[np.where(labels == anchor_label)[0]] = 0

    q_d_inv = torch.exp(log_q_d_inv - torch.max(log_q_d_inv))  # - max(log) for stability
    q_d_inv[np.where(labels == anchor_label)[0]] = 0

    # NOTE: Cutting of values with high distances made the results slightly worse. It can also lead to
    # errors where there are no available negatives (for high samples_per_class cases).
    # q_d_inv[np.where(d.detach().cpu().numpy()>self.upper_cutoff)[0]]    = 0

    q_d_inv = q_d_inv / q_d_inv.sum()
    return q_d_inv.detach().cpu().numpy()

import numpy as np

from sampling.utils import check_if_numpy, pair_wise_distance, inverse_sphere_distances


class SAMPLING:
    def __init__(self, opt):
        self.par = opt
        self.lower_cutoff = opt.miner_distance_lower_cutoff
        self.upper_cutoff = opt.miner_distance_upper_cutoff
        self.name = 'distance'

    def __call__(self, batch, labels, tar_labels=None, return_distances=False, distances=None):
        labels = check_if_numpy(labels)
        bs, dim = batch.shape

        if distances is None:
            distances = pair_wise_distance(batch.detach(), clamp_min=0).clamp(min=self.lower_cutoff)
        sel_d = distances.shape[-1]

        positives, negatives = [], []
        anchors = []

        tar_labels = labels if tar_labels is None else tar_labels

        for i in range(bs):
            pos = tar_labels == labels[i]

            anchors.append(i)
            # Sample negatives by distance
            q_d_inv = inverse_sphere_distances(dim, distances[i], tar_labels, labels[i])
            negatives.append(np.random.choice(sel_d, p=q_d_inv))

            # There must be one sample has the same label, e.g., self
            # consider remove the if condition to simplify the code
            if np.sum(pos) > 0:
                # Sample positives randomly
                # If there is more than one positive, exclude self
                if np.sum(pos) > 1:
                    pos[i] = 0
                positives.append(np.random.choice(np.where(pos)[0]))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]

        if return_distances:
            return sampled_triplets, distances
        else:
            return sampled_triplets

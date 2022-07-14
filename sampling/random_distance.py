import numpy as np

from sampling.utils import check_if_numpy, pair_wise_distance, inverse_sphere_distances


class SAMPLING:
    def __init__(self, opt):
        self.par = opt
        self.lower_cutoff = opt.miner_distance_lower_cutoff
        self.upper_cutoff = opt.miner_distance_upper_cutoff
        self.name = 'distance'

    def __call__(self, batch, labels):
        labels = check_if_numpy(labels)

        labels = labels[np.random.choice(len(labels), len(labels), replace=False)]

        bs, dim = batch.shape[0], batch.shape[1]
        distances = pair_wise_distance(batch.detach(), clamp_min=0).clamp(min=self.lower_cutoff)

        positives, negatives = [], []
        anchors = []

        for i in range(bs):
            pos = labels == labels[i]

            if np.sum(pos) > 1:
                anchors.append(i)
                q_d_inv = inverse_sphere_distances(dim, distances[i], labels, labels[i])
                # Sample positives randomly
                pos[i] = 0
                positives.append(np.random.choice(np.where(pos)[0]))
                # Sample negatives by distance
                negatives.append(np.random.choice(bs, p=q_d_inv))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        return sampled_triplets

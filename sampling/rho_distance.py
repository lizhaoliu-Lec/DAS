import numpy as np

from sampling.utils import check_if_numpy, pair_wise_distance, inverse_sphere_distances


class SAMPLING:
    def __init__(self, opt):
        self.par = opt
        self.lower_cutoff = opt.miner_rho_distance_lower_cutoff
        self.upper_cutoff = opt.miner_rho_distance_upper_cutoff
        self.contrastive_p = opt.miner_rho_distance_cp

        self.name = 'rho_distance'

    def __call__(self, batch, labels, return_distances=False):
        labels = check_if_numpy(labels)

        bs, dim = batch.shape[0], batch.shape[1]
        distances = pair_wise_distance(batch.detach(), clamp_min=1e-4).clamp(min=self.lower_cutoff)

        positives, negatives = [], []
        anchors = []

        for i in range(bs):
            pos = labels == labels[i]

            use_contr = np.random.choice(2, p=[1 - self.contrastive_p, self.contrastive_p])
            if np.sum(pos) > 1:
                anchors.append(i)
                if use_contr:
                    positives.append(i)
                    # Sample negatives by distance
                    pos[i] = 0
                    negatives.append(np.random.choice(np.where(pos)[0]))
                else:
                    q_d_inv = inverse_sphere_distances(dim, distances[i], labels, labels[i])
                    # Sample positives randomly
                    pos[i] = 0
                    positives.append(np.random.choice(np.where(pos)[0]))
                    # Sample negatives by distance
                    negatives.append(np.random.choice(bs, p=q_d_inv))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        self.push_triplets = np.sum([m[1] == m[2] for m in labels[sampled_triplets]])

        if return_distances:
            return sampled_triplets, distances
        else:
            return sampled_triplets

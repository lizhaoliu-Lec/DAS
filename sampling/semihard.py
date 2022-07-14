import numpy as np

from sampling.utils import check_if_numpy, pair_wise_distance


class SAMPLING:
    def __init__(self, opt):
        self.par = opt
        self.name = 'semihard'
        self.margin = vars(opt)['loss_' + opt.loss + '_margin']

    def __call__(self, batch, labels, return_distances=False):
        labels = check_if_numpy(labels)

        bs = batch.size(0)

        # Return distance matrix for all elements in batch (BSxBS)
        distances = pair_wise_distance(batch.detach(), clamp_min=0).detach().cpu().numpy()

        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            neg = labels != l
            pos = labels == l

            anchors.append(i)
            pos[i] = 0

            p = np.random.choice(np.where(pos)[0])
            positives.append(p)

            # Find negatives that violate triplet constraint semi-negatives
            # d_ap < d < d_ap + margin e.g., negative samples that within the margin
            neg_mask = np.logical_and(neg, d > d[p])
            neg_mask = np.logical_and(neg_mask, d < self.margin + d[p])

            if neg_mask.sum() > 0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg)[0]))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]

        if return_distances:
            return sampled_triplets, distances
        else:
            return sampled_triplets

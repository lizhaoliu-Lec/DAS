import numpy as np

from sampling.utils import check_if_numpy


class SAMPLING:
    def __init__(self, opt):
        self.par = opt
        self.name = 'npair'

    def __call__(self, batch, labels):
        labels = check_if_numpy(labels)

        anchors, positives, negatives = [], [], []

        for i in range(len(batch)):
            anchor = i
            pos = labels == labels[anchor]

            if np.sum(pos) > 1:
                anchors.append(anchor)
                avail_positive = np.where(pos)[0]
                avail_positive = avail_positive[avail_positive != anchor]
                positive = np.random.choice(avail_positive)
                positives.append(positive)

        negatives = []
        for anchor, positive in zip(anchors, positives):
            neg_idxs = [i for i in range(len(batch)) if i not in [anchor, positive] and labels[i] != labels[anchor]]
            # neg_idxs = [i for i in range(len(batch)) if i not in [anchor, positive]]
            negative_set = np.arange(len(batch))[neg_idxs]
            negatives.append(negative_set)

        return anchors, positives, negatives

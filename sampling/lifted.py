import numpy as np

from sampling.utils import check_if_numpy


class SAMPLING:
    def __init__(self, opt):
        self.par = opt
        self.name = 'lifted'

    def __call__(self, batch, labels):
        labels = check_if_numpy(labels)

        anchors, positives, negatives = [], [], []
        list(range(len(batch)))

        for i in range(len(batch)):
            anchor = i
            pos = labels == labels[anchor]

            if np.sum(pos) > 1:
                anchors.append(anchor)
                positive_set = np.where(pos)[0]
                positive_set = positive_set[positive_set != anchor]
                positives.append(positive_set)

        negatives = []
        for anchor, positive_set in zip(anchors, positives):
            neg_idxs = [i for i in range(len(batch)) if i not in [anchor] + list(positive_set)]
            negative_set = np.arange(len(batch))[neg_idxs]
            negatives.append(negative_set)

        return anchors, positives, negatives

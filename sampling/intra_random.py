import numpy as np

from sampling.utils import check_if_numpy


class SAMPLING:
    def __init__(self, opt):
        self.par = opt
        self.name = 'random'

    def __call__(self, batch, labels):
        labels = check_if_numpy(labels)

        unique_classes = np.unique(labels)
        indices = np.arange(len(batch))
        class_dict = {i: indices[labels == i] for i in unique_classes}

        sampled_triplets = []
        for cls in np.random.choice(list(class_dict.keys()), len(labels), replace=True):
            a, p, n = np.random.choice(class_dict[cls], 3, replace=True)
            sampled_triplets.append((a, p, n))

        return sampled_triplets

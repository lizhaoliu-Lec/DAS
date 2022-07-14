import numpy as np

from sampling.utils import check_if_numpy, pair_wise_distance


class SAMPLING:
    def __init__(self, opt):
        self.par = opt
        self.mode = opt.miner_parametric_mode
        self.n_support = opt.miner_parametric_n_support
        self.support_lim = opt.miner_parametric_support_lim
        self.name = 'parametric'

        self.sample_distribution = self.get_sample_distribution()

        self.support = None

    def __call__(self, batch, labels):
        bs = batch.shape[0]
        sample_distribution = self.sample_distribution

        labels = check_if_numpy(labels)

        distances = pair_wise_distance(batch.detach(), clamp_min=1e-4)

        p_assigns = np.sum((distances.cpu().numpy().reshape(-1) > self.support[1:-1].reshape(-1, 1)).T, axis=1).reshape(
            distances.shape)
        outside_support_lim = (distances.cpu().numpy().reshape(-1) < self.support_lim[0]) * (
                distances.cpu().numpy().reshape(-1) > self.support_lim[1])
        outside_support_lim = outside_support_lim.reshape(distances.shape)

        sample_ps = sample_distribution[p_assigns]
        sample_ps[outside_support_lim] = 0

        anchors, labels_visited = [], []
        positives, negatives = [], []

        for i in range(bs):
            neg = labels != labels[i]
            pos = labels == labels[i]

            if np.sum(pos) > 1:
                anchors.append(i)

                # Sample positives randomly
                pos[i] = 0
                positives.append(np.random.choice(np.where(pos)[0]))

                # Sample negatives by distance
                sample_p = sample_ps[i][neg]
                sample_p = sample_p / sample_p.sum()
                negatives.append(np.random.choice(np.arange(bs)[neg], p=sample_p))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        return sampled_triplets

    def get_sample_distribution(self):
        self.support = np.linspace(self.support_lim[0], self.support_lim[1], self.n_support)

        if self.mode == 'uniform':
            sample_distribution = np.array([1.] * (self.n_support - 1))

        elif self.mode == 'hards':
            sample_distribution = self.support.copy()
            sample_distribution[self.support <= 0.5] = 1
            sample_distribution[self.support > 0.5] = 0

        elif self.mode == 'semihards':
            sample_distribution = self.support.copy()
            from IPython import embed
            embed()
            sample_distribution[(self.support <= 0.7) * (self.support >= 0.3)] = 1
            sample_distribution[(self.support < 0.3) * (self.support > 0.7)] = 0

        elif self.mode == 'veryhards':
            sample_distribution = self.support.copy()
            sample_distribution[self.support <= 0.3] = 1
            sample_distribution[self.support > 0.3] = 0

        else:
            raise ValueError('Unsupported mode: `%s`' % self.mode)

        sample_distribution = np.clip(sample_distribution, 1e-15, 1)
        sample_distribution = sample_distribution / sample_distribution.sum()
        return sample_distribution

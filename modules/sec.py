class SEC:
    def __init__(self, momentum=0.999, weight=0.01):
        self.momentum = momentum
        self.weight = weight
        self.norm_mean = None

    def __call__(self, features):
        fea_norm = features.norm(p=2, dim=1)
        self.update_norm_mean(fea_norm.mean().detach())
        sec_loss = ((fea_norm - self.norm_mean) ** 2).mean()

        return sec_loss

    def update_norm_mean(self, new_norm_mean):
        if self.norm_mean is None:
            self.norm_mean = new_norm_mean
        else:
            m = self.momentum
            self.norm_mean = m * self.norm_mean + (1 - m) * new_norm_mean

import torch
import torch.nn as nn
import torch.nn.functional as F

ALLOWED_MINING_OPS = None
REQUIRES_SAMPLING = False
REQUIRES_OPTIM = True


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Criterion(nn.Module):
    def __init__(self, opt):
        """
        Args:
            opt: Namespace containing all relevant parameters.
        """
        super(Criterion, self).__init__()
        nb_classes = opt.n_classes
        sz_embed = opt.embed_dim
        mrg = opt.loss_proxyanchor_margin  # set to 0.1 by default
        alpha = opt.loss_proxyanchor_alpha  # set to 32 by default
        decay = opt.loss_proxyanchor_decay  # set to 0.0 by default

        # nb_classes, sz_embed, mrg = 0.1, alpha = 32
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.name = 'proxyanchor'

        # opt.loss_proxyanchor_lrmulti set to 100 by default
        self.optim_dict_list = [{'params': self.proxies,
                                 'lr': opt.lr * opt.loss_proxyanchor_lrmulti,
                                 'weight_decay': opt.loss_proxyanchor_decay}]

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_SAMPLING = REQUIRES_SAMPLING
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

        self.par = opt

    def forward(self, batch, labels, **kwargs):
        P = self.proxies

        cos = F.linear(l2_norm(batch), l2_norm(P))  # calculate cosine similarity
        P_one_hot = binarize(T=labels, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        positive_cos = cos

        pos_exp = torch.exp(-self.alpha * (positive_cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss

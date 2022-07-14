from loss import adversarial_separation
from loss import angular, snr, histogram, arcface
from loss import lifted, contrastive, softmax
from loss import softtriplet, multisimilarity, quadruplet
from loss import triplet, margin, proxynca, npair, proxyanchor


def select(loss, opt, sampling=None):
    losses = {'triplet': triplet,
              'margin': margin,
              'proxynca': proxynca,
              'proxyanchor': proxyanchor,
              'npair': npair,
              'angular': angular,
              'contrastive': contrastive,
              'lifted': lifted,
              'snr': snr,
              'multisimilarity': multisimilarity,
              'histogram': histogram,
              'softmax': softmax,
              'softtriplet': softtriplet,
              'arcface': arcface,
              'quadruplet': quadruplet,
              'adversarial_separation': adversarial_separation}

    if loss not in losses:
        raise NotImplementedError('Loss {} not implemented!'.format(loss))

    loss_lib = losses[loss]
    if loss_lib.REQUIRES_SAMPLING:
        if sampling is None:
            raise Exception('Loss {} requires one of the following batch mining methods: {}'.format(loss,
                                                                                                    loss_lib.ALLOWED_MINING_OPS))
        else:
            if sampling.name not in loss_lib.ALLOWED_MINING_OPS:
                raise Exception('{}-mining not allowed for {}-loss!'.format(sampling.name, loss))

    loss_par_dict = {'opt': opt}
    if loss_lib.REQUIRES_SAMPLING:
        loss_par_dict['sampling'] = sampling

    criterion = loss_lib.Criterion(**loss_par_dict)

    to_optim = None
    if loss_lib.REQUIRES_OPTIM:
        if hasattr(criterion, 'optim_dict_list') and criterion.optim_dict_list is not None:
            to_optim = criterion.optim_dict_list
        else:
            to_optim = [{'params': criterion.parameters(), 'lr': criterion.lr}]

    return criterion, to_optim, loss_lib

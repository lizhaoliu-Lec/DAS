from sampling import lifted, rho_distance, softhard, npair, parametric, random, semihard, distance
from sampling import random_distance, intra_random

BATCHMINING_METHODS = {'random': random,
                       'semihard': semihard,
                       'softhard': softhard,
                       'distance': distance,
                       'rho_distance': rho_distance,
                       'npair': npair,
                       'parametric': parametric,
                       'lifted': lifted,
                       'random_distance': random_distance,
                       'intra_random': intra_random}


def select(samplingname, opt):
    if samplingname not in BATCHMINING_METHODS:
        raise NotImplementedError('Batchmining {} not available!'.format(samplingname))

    batchmine_lib = BATCHMINING_METHODS[samplingname]

    return batchmine_lib.SAMPLING(opt)

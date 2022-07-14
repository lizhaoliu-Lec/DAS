import torch


def triplet_cnt(triplet, bs):
    real_cnt, pseudo_cnt = 0, 0
    for t in triplet:
        if t > bs:
            pseudo_cnt += 1
        else:
            real_cnt += 1
    assert (real_cnt + pseudo_cnt) == 3, 'real_cnt + pseudo_cnt = {}'.format(real_cnt + pseudo_cnt)
    return real_cnt, pseudo_cnt


def pair_cnt(pair, bs):
    real_cnt, pseudo_cnt = 0, 0
    for p in pair:
        if p > bs:
            pseudo_cnt += 1
        else:
            real_cnt += 1
    assert (real_cnt + pseudo_cnt) == 2, 'real_cnt + pseudo_cnt = {}'.format(real_cnt + pseudo_cnt)
    return real_cnt, pseudo_cnt


def bl_triplets_weight(triplets, bs, device=None):
    weight = []
    for triplet in triplets:
        num_real, num_pseudo = triplet_cnt(triplet, bs)
        weight.append(num_real / (num_real + num_pseudo))
    weight = torch.tensor(weight, requires_grad=False)
    if device is not None:
        weight = weight.to(device=device)
    return weight


def bl_pairs_weight(pairs, bs, device=None):
    weight = []
    for pair in pairs:
        num_real, num_pseudo = pair_cnt(pair, bs)
        weight.append(num_real / num_pseudo)
    weight = torch.tensor(weight, requires_grad=False)
    if device is not None:
        weight = weight.to(device=device)
    return weight


def bl_proxy_weight(num_embeddings, bs, z=0.2, device=None):
    weight = [1. for _ in range(bs)]
    weight.extend([z * bs / (num_embeddings - bs) for _ in range(num_embeddings - bs)])
    weight = torch.tensor(weight, requires_grad=False)
    if device is not None:
        weight = weight.to(device=device)
    return weight


if __name__ == '__main__':
    ret = bl_proxy_weight(336, 112)
    print("===> proxy_ret: {}".format(ret))

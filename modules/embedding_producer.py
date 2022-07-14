import random

import torch
import torch.nn.functional as F


def l2_dist(x1, x2):
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=-1))


def range_rand(left, right, shape, device):
    """
    produce random within range [left, right).
    """
    return (right - left) * torch.rand(shape, device=device) + left


class DenselyAnchoredSampling:
    def __init__(self,
                 num_classes, dim,
                 num_produce=3, normalize=True,
                 dfs_num_scale=4, dfs_scale_range=(0.5, 2.0),
                 mts_num_transformation_bank=10, mts_scale=0.01,
                 detach=True):

        # general parameters
        self.num_classes = num_classes
        self.dim = dim

        # parameters for DAS
        self.num_produce = num_produce
        self.normalize = normalize

        # parameters for DFS (Discriminative Feature Scaling)
        self.dfs_num_scale = dfs_num_scale
        self.dfs_scale_range = dfs_scale_range
        self.frequency_recorder_matrix = {k: [0 for _ in range(dim)] for k in range(num_classes)}

        # parameters for MTS (Memorized Transformation Shifting)
        self.mts_num_transformation_bank = mts_num_transformation_bank
        self.mts_scale = mts_scale
        self.class_banks = [
            MemoryBank(queue_size=self.mts_num_transformation_bank, queue_dim=dim) for _ in range(num_classes)
        ]

        self.detach = detach

    def __call__(self, embeddings, targets):
        # update frequency recorder matrix
        self._update_frequency_recorder_matrix(embeddings, targets)

        # update transformation bank
        self._update_class_bank(embeddings, targets)

        # produce the embedding
        produced_embeddings, produced_targets = self._produce(embeddings, targets)

        return produced_embeddings, produced_targets

    @torch.no_grad()
    def _update_frequency_recorder_matrix(self, embeddings, targets):
        for embedding, target in zip(embeddings, targets):
            _, topK_index = torch.topk(embedding, k=self.dfs_num_scale,
                                       dim=0, largest=True)
            for ind in topK_index:
                self.frequency_recorder_matrix[target.item()][ind.item()] += 1

    def _produce(self, embeddings, targets):
        device = embeddings.device
        produced_embeddings, produced_targets = [], []

        for _ in range(self.num_produce):
            # get topK index
            topK_index = []
            for target in targets:
                _, _topK_index = torch.topk(torch.Tensor(self.frequency_recorder_matrix[target.item()]),
                                            k=self.dfs_num_scale, dim=0, largest=True)
                topK_index.append(_topK_index)
            topK_index = torch.stack(topK_index, dim=0)
            topK_index = topK_index.to(device=device)

            # obtain semantic scaling factors
            scale = range_rand(self.dfs_scale_range[0], self.dfs_scale_range[1],
                               shape=(embeddings.size(0), self.dfs_num_scale), device=device)

            # obtain semantic shifting factors
            bias = []
            for _topK_index, target in zip(topK_index, targets):
                transformations, _ = self.class_banks[target].get()

                if transformations.size(0) > 1:
                    rand_index = random.randint(0, transformations.size(0) - 1)
                else:
                    rand_index = 0

                if transformations.size(0) == 0:
                    _bias = torch.zeros((embeddings.size(-1),))
                else:
                    _bias = transformations[rand_index, :]
                _bias = _bias.to(device=device)
                bias.append(_bias)
            bias = torch.stack(bias, dim=0)
            bias = self.mts_scale * bias.to(device=device)

            if not self.detach:
                _produced_embeddings = embeddings.clone()
            else:
                _produced_embeddings = embeddings.clone().detach()

            # perform semantic scaling
            # only operate on best channels
            # (1) first retrieve the values by gather according to topK_index
            _part_generated_embeddings = _produced_embeddings.gather(dim=1, index=topK_index)
            # print("===> _part_generated_embeddings.size(): {}".format(_part_generated_embeddings.size()))
            # (2) then perform affine on it
            _part_generated_embeddings = _part_generated_embeddings * scale
            # (3) finally, put the values to where they belong
            _produced_embeddings = _produced_embeddings.scatter(dim=1, index=topK_index,
                                                                src=_part_generated_embeddings)

            # perform semantic shifting
            _produced_embeddings += bias

            # if not normalize, then we will not perform normalization
            # this is for the loss such as, n-pair, lifted that do not normalize the embeddings
            if self.normalize:
                _produced_embeddings = F.normalize(_produced_embeddings, dim=1)

            produced_embeddings.append(_produced_embeddings)
            produced_targets.append(targets.clone())

        return produced_embeddings, produced_targets

    def _update_class_bank(self, embeddings, targets):
        # update intra class banks
        detached_embeddings = embeddings.clone().detach()  # (N, dim)
        unique_targets = torch.unique(targets)
        for t in unique_targets:
            # detach the previous transformation before updating
            self.class_banks[t].detach()

            # retrieve the embeddings of class t
            target_embeddings = detached_embeddings[targets == t]  # dim: (M, dim)

            transformation = target_embeddings[:, None, ...] - target_embeddings[None, ...]
            transformation = transformation.reshape([-1, self.dim])
            filter_index = torch.sum(torch.abs(transformation), dim=1)
            transformation = transformation[filter_index > 0]
            # print("update class ", t)
            self.class_banks[t].enqueue_dequeue(transformation,
                                                torch.zeros(transformation.size(0)) + t)

    def _get_pairwise_transformation(self, _embeddings_a, _embeddings_b):
        _transformation = _embeddings_a[:, None, ...] - _embeddings_b[None, ...]
        _transformation = _transformation.reshape([-1, self.dim])
        _filter_index = torch.sum(torch.abs(_transformation), dim=1)
        _transformation = _transformation[_filter_index > 0]
        return _transformation


class MemoryBank:
    """
    Queue for Memorized Transformation Shifting (MTS).
    Note that we disentangle its implementation from MTS
    for more flexible usage such as two queues (One for inter-class and one for intra-class) in MTS.
    """

    def __init__(self, queue_size, queue_dim):
        self.queue_size = queue_size
        self.queue_dim = queue_dim
        self.ptr = 0
        self.transformations = torch.zeros(queue_size, queue_dim).cuda()
        self.targets = torch.zeros(queue_size, dtype=torch.long)

    @property
    def is_full(self):
        return self.targets[-1].item() != 0

    @property
    def size(self):
        if self.is_full:
            return self.queue_size
        else:
            return self.ptr

    @property
    def norm(self):
        with torch.no_grad():
            curr_trans = self.get()[0]  # dim: (N, d)
            if curr_trans.size(0) == 0:
                return 0

            trans_norm = torch.norm(curr_trans, p=2, dim=1)  # dim: (N, 1)

            avg_norm = torch.mean(trans_norm)  # dim: (1)

            return avg_norm.item()

    def get(self):
        filter_index = torch.sum(torch.abs(self.transformations), dim=1) > 0
        return self.transformations[filter_index], self.targets[filter_index]

    def enqueue_dequeue(self, trans, targets):
        q_size = len(targets)
        # when the queue size is smaller than current batch size
        # we just get the remained part of current batch
        if q_size > self.queue_size:
            self.transformations = trans[-self.queue_size:]
            self.targets = targets[-self.queue_size:]
        elif self.ptr + q_size > self.queue_size:
            # if the remaining size is smaller than current batch size
            # things become a little difficult
            # we address this by partition the current batch into the end part and the beginning part

            # obtain the beginning part size
            end_ptr = min(self.ptr + q_size, self.queue_size)
            remain = q_size - (end_ptr - self.ptr)

            # update the end part
            self.transformations[self.ptr:end_ptr] = trans[:-remain]
            self.targets[self.ptr:end_ptr] = targets[:-remain]

            # update the beginning part
            self.transformations[:remain] = trans[-remain:]
            self.targets[:remain] = targets[-remain:]

            # update the ptr
            self.ptr = remain
        else:
            self.transformations[self.ptr: self.ptr + q_size] = trans
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size

    def detach(self):
        self.transformations = self.transformations.detach()


class EmbeddingProducer:
    def __init__(self,
                 num_classes, dim, produce_type='DAS',
                 # for das
                 das_num_produce=3, das_normalize=False,
                 # for dfs
                 das_dfs_num_scale=4, das_dfs_scale_left=0.5, das_dfs_scale_right=2.0,
                 # for mts
                 das_mts_num_transformation_bank=10, das_mts_scale=0.01,
                 aux_loss=False, aux_loss_weight=1.0, detach=True):
        supported_retrieve_type = ['DAS']
        assert produce_type in supported_retrieve_type

        self.num_classes = num_classes
        self.dim = dim
        self.produce_type = produce_type

        # DAS instance
        self.densely_anchored_sampling = None

        # parameters for DAS
        self.das_num_produce = None
        self.das_normalize = None

        # parameters for DFS (Discriminative Feature Scaling)
        self.das_dfs_num_scale = None
        self.das_dfs_scale_left = None
        self.das_dfs_scale_right = None

        # parameters for MTS (Memorized Transformation Shifting)
        self.das_mts_num_transformation_bank = None
        self.das_mts_scale = None

        # record the distance of the produced embeddings and original embeddings
        # to simplify the implementation, we restrict the format of final embeddings as
        # [batch_embeddings, batch_produced_embeddings, ..., batch_produced_embeddings]
        # where batch_embeddings.size() == batch_produced_embeddings.size() always hold true
        self._embedding_distance = 0
        # for embedding that has not been normalized
        # we also record the distance between the embedding and the generated ooe
        self._embedding_distance_after_norm = 0
        self._num_produced = 0
        self._num_real = 0
        self.aux_loss = aux_loss
        self.aux_loss_weight = aux_loss_weight
        self.detach = detach

        if self.produce_type == 'DAS':
            # DAS
            self.das_num_produce = das_num_produce
            self.das_normalize = das_normalize

            # DFS
            self.das_dfs_num_scale = das_dfs_num_scale
            self.das_dfs_scale_left = das_dfs_scale_left
            self.das_dfs_scale_right = das_dfs_scale_right

            # MTS
            self.das_mts_num_transformation_bank = das_mts_num_transformation_bank
            self.das_mts_scale = das_mts_scale

            self.densely_anchored_sampling = DenselyAnchoredSampling(
                num_classes=self.num_classes, dim=self.dim,
                num_produce=self.das_num_produce, normalize=self.das_normalize,
                dfs_num_scale=self.das_dfs_num_scale,
                dfs_scale_range=(self.das_dfs_scale_left, self.das_dfs_scale_right),
                mts_num_transformation_bank=self.das_mts_num_transformation_bank, mts_scale=self.das_mts_scale,
                detach=self.detach)

    def __call__(self, embeddings, labels):

        self._num_real = embeddings.size(0)

        # produce embeddings (along with labels)
        produced_embeddings, produced_labels = self._produce(embeddings.clone(), labels.clone(),
                                                             produce_type=self.produce_type)

        # produce and combine the embeddings
        if produced_embeddings is not None:
            produced_embeddings = produced_embeddings.to(embeddings.device)
            produced_labels = produced_labels.to(labels.device)

            combined_embeddings = torch.cat([embeddings, produced_embeddings])
            combined_labels = torch.cat([labels, produced_labels])

            # print("===> combined_embeddings.size(): {}".format(combined_embeddings.size()))
            # print("===> combined_labels.size(): {}".format(combined_labels.size()))
            # print("===> produced_embeddings.size(): {}".format(produced_embeddings.size()))
            # print("===> produced_labels.size(): {}".format(produced_labels.size()))
            self._num_produced = combined_embeddings.size(0) - self._num_real
            return combined_embeddings, combined_labels
        else:
            self._num_produced = embeddings.size(0) - self._num_real
            return embeddings, labels

    def _produce(self, embeddings, labels, produce_type):
        if produce_type == 'DAS':
            produced_embeddings, produced_targets = self.densely_anchored_sampling(embeddings, labels)
            self.calculate_distance(embeddings, produced_embeddings)
            produced_embeddings = torch.cat(produced_embeddings, dim=0)
            produced_targets = torch.cat(produced_targets, dim=0)
            return produced_embeddings, produced_targets
        else:
            raise NotImplementedError('Unsupported retrieve type `{}`'.format(produce_type))

    def __repr__(self):
        return "EmbeddingProducer(" \
               "\n\tnum_classes={}, produce_type={}," \
               "\n\tDenselyAnchoredSampling(" \
               "\n\tdas_num_produce={}, das_normalize={}, " \
               "\n\tdas_dfs_num_scale={}, das_dfs_scale_left={}, das_dfs_scale_right={}, " \
               "\n\tdas_mts_num_transformation_bank={}, das_mts_scale={}, " \
               "\n\t)" \
               "\n)".format(
            self.num_classes, self.produce_type,
            # for das
            self.das_num_produce, self.das_normalize,
            # for dfs
            self.das_dfs_num_scale, self.das_dfs_scale_left, self.das_dfs_scale_right,
            # for mts
            self.das_mts_num_transformation_bank, self.das_mts_scale,
        )

    @property
    def embedding_distance(self):
        return self._embedding_distance

    @property
    def embedding_distance_after_norm(self):
        return self._embedding_distance_after_norm

    @torch.no_grad()
    def calculate_distance(self, embeddings, pseudo_embeddings):
        # embeddings: (b, dim)
        # pseudo_embeddings: [(b, dim), ..., (b, dim)]
        # reset the embedding distance
        self._embedding_distance = 0
        dis_cnt = 0
        for _pseudo_embeddings in pseudo_embeddings:
            batch_dis = l2_dist(embeddings, _pseudo_embeddings)
            # print("===> batch_dis.size(): {}".format(batch_dis))
            self._embedding_distance += torch.sum(batch_dis).item()
            dis_cnt += batch_dis.size(0)

        if dis_cnt >= 1:
            self._embedding_distance /= dis_cnt
        else:
            self._embedding_distance = 0

    @torch.no_grad()
    def calculate_embedding_distance_after_norm(self, embeddings, pseudo_embeddings):
        # embeddings: (b, dim)
        # pseudo_embeddings: [(b, dim), ..., (b, dim)]
        # reset the embedding distance after norm
        self._embedding_distance_after_norm = 0
        dis_cnt = 0
        for _pseudo_embeddings in pseudo_embeddings:
            batch_dis = l2_dist(embeddings, _pseudo_embeddings)
            # print("===> batch_dis.size(): {}".format(batch_dis))
            self._embedding_distance_after_norm += torch.sum(batch_dis).item()
            dis_cnt += batch_dis.size(0)

        if dis_cnt >= 1:
            self._embedding_distance_after_norm /= dis_cnt
        else:
            self._embedding_distance_after_norm = 0

    @property
    def num_real(self):
        return self._num_real

    @property
    def num_produced(self):
        return self._num_produced

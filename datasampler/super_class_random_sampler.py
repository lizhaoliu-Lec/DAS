import random

import copy
import torch

REQUIRES_STORAGE = False


class Sampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package with super class.
    """

    def __init__(self, opt, image_dict, image_list):
        self.pars = opt

        #####
        self.image_dict = image_dict
        self.image_list = image_list

        # for super class
        class_ix2super_ix = opt.class_ix2super_ix
        self.class_ix2super_ix = class_ix2super_ix
        self.super_classes = list(set(class_ix2super_ix.values()))
        self.super_ix2class_ixs = {
            _: [] for _ in self.super_classes
        }
        for k, v in self.class_ix2super_ix.items():
            self.super_ix2class_ixs[v].append(k)

        num_super = opt.data_num_super
        assert num_super <= len(self.super_classes)
        self.num_super = num_super

        #####
        self.classes = list(self.image_dict.keys())

        ####
        self.batch_size = opt.bs  # 112
        self.samples_per_class = opt.samples_per_class  # 2
        self.sampler_length = len(image_list) // opt.bs  # num_class / bs
        assert self.batch_size % self.samples_per_class == 0, '#Samples per class must divide batchsize!'

        self.name = 'super_class_random_sampler'
        self.requires_storage = False

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            # Random Subset from Random classes
            draws = self.batch_size // self.samples_per_class
            # randomly select #num_super super classes
            super_classes = copy.deepcopy(self.super_classes)
            random.shuffle(super_classes)
            classes = []
            for super_class in super_classes[:self.num_super]:
                classes.extend(self.super_ix2class_ixs[super_class])

            for _ in range(draws):
                class_key = random.choice(classes)
                class_ix_list = [random.choice(self.image_dict[class_key])[-1] for _ in range(self.samples_per_class)]
                subset.extend(class_ix_list)

            yield subset

    def __len__(self):
        return self.sampler_length

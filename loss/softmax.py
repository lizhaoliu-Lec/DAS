import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss._share import bl_proxy_weight

ALLOWED_MINING_OPS = None
REQUIRES_SAMPLING = False
REQUIRES_OPTIM = True


class Criterion(nn.Module):
    """
    This Implementation follows: https://github.com/azgo14/classification_metric_learning
    """

    def __init__(self, opt):
        super(Criterion, self).__init__()
        self.par = opt
        self.opt = opt

        self.temperature = opt.loss_softmax_temperature

        self.class_map = nn.Parameter(torch.Tensor(opt.n_classes, opt.embed_dim))
        stdv = 1. / np.sqrt(self.class_map.size(1))
        self.class_map.data.uniform_(-stdv, stdv)

        self.name = 'softmax'

        self.lr = opt.loss_softmax_lr

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_SAMPLING = REQUIRES_SAMPLING
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

        # flag to denote whether to use concat operation on self.class_map
        self.low_weight = None
        self.high_weight_1 = None
        self.high_weight_2 = None
        self.concat_type = None

    def forward(self, batch, labels, **kwargs):
        if self.low_weight is not None and (self.high_weight_1 is not None or self.high_weight_2 is not None):
            runtime_class_map = self.get_runtime_class_map()
            # print("====> runtime_class_map.size() ", runtime_class_map.size())
            # print("====> runtime_class_map.device ", runtime_class_map.device)
            class_mapped_batch = F.linear(batch, F.normalize(runtime_class_map, dim=1))
        else:
            class_mapped_batch = F.linear(batch, F.normalize(self.class_map, dim=1))

        loss = nn.CrossEntropyLoss()(class_mapped_batch / self.temperature,
                                         labels.to(torch.long).to(self.par.device))

        return loss

    def get_runtime_class_map(self):
        if self.concat_type == 'left':
            return torch.cat([self.low_weight, self.high_weight_1], dim=1)
        elif self.concat_type == 'right':
            return torch.cat([self.high_weight_1, self.low_weight], dim=1)
        elif self.concat_type == 'middle':
            return torch.cat([self.high_weight_1, self.low_weight, self.high_weight_2], dim=1)
        else:
            raise ValueError('Unexpected concat_type `%s` occur' % self.concat_type)

    def load_low_dimensional_classifier(self, low_cls, start_index, end_index, freeze):

        low_dim = low_cls.class_map.size(1)
        high_dim = self.class_map.size(1)

        assert end_index <= high_dim, 'end_index (%d) > high_dim (%d)' % (low_dim, high_dim)

        assert low_dim < high_dim, 'low_dim (%d) >= high_dim (%d)' % (low_dim, high_dim)

        assert low_dim == (end_index - start_index), 'low_dim (%d) != (end_index - start_index) (%d)' % (
            low_dim, (end_index - start_index))

        opt = self.par
        num_class = opt.n_classes
        dim = opt.embed_dim
        self.low_weight = nn.Parameter(low_cls.class_map.data, requires_grad=not freeze)

        # three type

        # low cls in the left
        if start_index == 0:
            self.concat_type = 'left'
            self.high_weight_1 = nn.Parameter(torch.Tensor(num_class, dim - end_index), requires_grad=True)
        # low cls in the right
        elif end_index == dim:
            self.concat_type = 'right'
            self.high_weight_1 = nn.Parameter(torch.Tensor(num_class, start_index), requires_grad=True)
        # low cls in the middle
        else:
            self.concat_type = 'middle'
            self.high_weight_1 = nn.Parameter(torch.Tensor(num_class, start_index), requires_grad=True)
            self.high_weight_2 = nn.Parameter(torch.Tensor(num_class, dim - end_index), requires_grad=True)


if __name__ == '__main__':
    def run_low_cls():
        import argparse
        def get_parameters():
            parser = argparse.ArgumentParser()

            parser.add_argument('--loss_softmax_temperature', default=0.05, type=float,
                                help='Temperature for NCA objective.')
            parser.add_argument('--embed_dim', default=128, type=int,
                                help='Embedding dimensionality of the network. '
                                     'Note: dim = 64, 128 or 512 is used in most papers, depending on the architecture.')
            parser.add_argument('--device', default=0, type=int, help='Gpu to use.')

            return parser

        opt = get_parameters().parse_args()
        opt.n_classes = 120
        opt.embed_dim = 64
        low_cls = Criterion(opt)
        opt.embed_dim = 128
        high_cls = Criterion(opt)

        # high_cls.load_low_dimensional_classifier(low_cls, 0, 64, True)
        # high_cls.load_low_dimensional_classifier(low_cls, 64, 128, True)
        high_cls.load_low_dimensional_classifier(low_cls, 32, 96, True)

        for name, param in high_cls.named_parameters():
            print(
                "====> name: %s, param.require_grad: %s, param.size(): %s" % (name, param.requires_grad, param.size()))

        print("======> high_cls.high_weight_1.grad: ", high_cls.high_weight_1.grad)
        print("======> high_cls.high_weight_2.grad: ", high_cls.high_weight_2.grad)
        print("======> high_cls.low_weight.grad: ", high_cls.low_weight.grad)

        high_cls.cuda()

        feat = torch.randn((5, 128)).cuda()
        label = torch.randint(0, 120, (5,)).cuda()
        print("====> label: ", label)
        loss = high_cls(feat, label)

        print("=====> loss: ", loss)

        loss.backward()

        print("=====> high_cls.high_weight_1.grad: ", high_cls.high_weight_1.grad)
        print("=====> high_cls.high_weight_2.grad: ", high_cls.high_weight_2.grad)
        print("=====> high_cls.low_weight.grad: ", high_cls.low_weight.grad)


    run_low_cls()

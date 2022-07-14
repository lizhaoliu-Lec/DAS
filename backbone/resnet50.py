"""
The network backbone and weights are adapted and used from the great
https://github.com/Cadene/pretrained-models.pytorch.
"""
import pretrainedmodels as ptm
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars = opt
        self.model = ptm.__dict__['resnet50'](num_classes=1000,
                                              pretrained='imagenet' if not opt.not_pretrained else None)

        self.name = opt.arch

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, opt.embed_dim)

        if 'bn_norm' in opt.arch:
            print("Using BN Norm after the final linear!!!")
            self.bn = nn.BatchNorm1d(num_features=opt.embed_dim)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])
        self.pool_base = nn.AdaptiveAvgPool2d(1)
        self.pool_aux = nn.AdaptiveMaxPool2d(1) if 'double' in opt.arch else None

        self.out_adjust = None
        self.x_before_normed = None

    def forward(self, x, warmup=False, **kwargs):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        # ckpt here, record x_before_pooled
        x_before_pooled = x
        x_pooled = self.pool_base(x)
        if self.pool_aux is not None:
            x_pooled += self.pool_aux(x_before_pooled)
        if warmup:
            x_pooled, x = x_pooled.detach(), x.detach()
        if self.pars.drop > 0 and self.training:
            x_pooled = F.dropout(x_pooled, p=self.pars.drop)
        x = self.model.last_linear(x_pooled.view(x.size(0), -1))
        # ckpt here, record x_before_normed
        self.x_before_normed = x
        if 'normalize' in self.pars.arch:
            x = nn.functional.normalize(x, dim=-1)
        if self.out_adjust and not self.train:
            x = self.out_adjust(x)

        return x, (x_pooled, x_before_pooled)

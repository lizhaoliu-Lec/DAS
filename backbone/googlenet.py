"""
The network architectures and weights are adapted and used from the great
https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as mod


class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars = opt
        self.model = mod.googlenet(pretrained=True)

        self.model.last_linear = nn.Linear(self.model.fc.in_features, opt.embed_dim)
        self.model.fc = None

        self.pool_base = nn.AdaptiveAvgPool2d(1)
        self.pool_aux = nn.AdaptiveMaxPool2d(1) if 'double' in opt.arch else None

        self.name = opt.arch

        self.transform_input = self.model.transform_input

    def forward(self, x, warmup=False, **kwargs):
        x_before_pooled = self.base_forward(x)
        x_pooled = self.pool_base(x_before_pooled)
        if self.pool_aux is not None:
            x_pooled += self.pool_aux(x_before_pooled)
        if warmup:
            x_pooled, x = x_pooled.detach(), x.detach()
        if self.pars.drop > 0 and self.training:
            x_pooled = F.dropout(x_pooled, p=self.pars.drop)
        x = self.model.last_linear(torch.reshape(x_pooled, (x.size(0), -1)))
        if 'normalize' in self.pars.arch:
            x = F.normalize(x, dim=-1)
        return x, (x_pooled, x_before_pooled)

    def base_forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        x = self.model.conv1(x)
        x = self.model.maxpool1(x)
        x = self.model.conv2(x)
        x = self.model.conv3(x)
        x = self.model.maxpool2(x)

        x = self.model.inception3a(x)
        x = self.model.inception3b(x)
        x = self.model.maxpool3(x)
        x = self.model.inception4a(x)

        x = self.model.inception4b(x)
        x = self.model.inception4c(x)
        x = self.model.inception4d(x)

        x = self.model.inception4e(x)
        x = self.model.maxpool4(x)
        x = self.model.inception5a(x)
        x = self.model.inception5b(x)

        return x

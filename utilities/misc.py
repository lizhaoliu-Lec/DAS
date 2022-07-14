import numpy as np
import torch.nn as nn


def gimme_params(model):
    """
    ACQUIRE NUMBER OF WEIGHTS
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def gimme_save_string(opt):
    """
    SAVE TRAINING PARAMETERS IN NICE STRING
    """
    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key], dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t' + str(sub_key) + ': ' + str(sub_item)
        else:
            base_str += '\n\t' + str(varx[key])
        base_str += '\n\n'
    return base_str


class DataParallel(nn.Module):
    def __init__(self, model, device_ids, dim):
        super().__init__()
        self.model = model.model
        self.network = nn.DataParallel(model, device_ids, dim)

    def forward(self, x):
        return self.network(x)

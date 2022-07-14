from scripts.parameters import read_arguments_from_cmd
from scripts import standard

import torch
import matplotlib
import warnings

if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')

    matplotlib.use('agg')
    warnings.filterwarnings("ignore")

    par = read_arguments_from_cmd()

    training_type = par.training_script

    print("** Training with `training_type = %s` **" % training_type)

    if training_type == 'standard':
        standard.main(par)
    else:
        raise ValueError('Unsupported training type `{}`'.format(training_type))

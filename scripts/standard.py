import os
import time

import numpy as np
import torch.multiprocessing

import backbone as archs
import sampling as bmine
import loss as criteria
import metric
from modules.embedding_producer import EmbeddingProducer
from modules.sec import SEC
from scripts._share import set_seed, get_dataloaders, train_one_epoch, evaluate
from utilities import logger
from utilities import misc


def main(opt):
    opt.source_path += '/' + opt.dataset
    opt.save_path += '/' + opt.dataset

    # The following setting is useful when logging to wandb and running multiple seeds per setup:
    # By setting the savename to <group_plus_seed>, the savename will instead comprise the group and the seed!
    if opt.savename == 'group_plus_seed':
        opt.savename = ''

    full_training_start_time = time.time()

    # Assert that the construction of the batch makes sense, i.e. the division into class-subclusters.
    assert not opt.bs % opt.samples_per_class, \
        'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'

    opt.pretrained = not opt.not_pretrained

    # GPU SETTINGS
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # if not opt.use_data_parallel:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu[0])

    # SEEDS FOR REPRODUCIBILITY.
    set_seed(opt.seed)

    # NETWORK SETUP
    opt.device = torch.device('cuda')
    model = archs.select(opt.arch, opt)

    if opt.fc_lr < 0:
        to_optim = [{'params': model.parameters(), 'lr': opt.lr, 'weight_decay': opt.decay}]
    else:
        all_but_fc_params = [x[-1] for x in list(filter(lambda x: 'last_linear' not in x[0], model.named_parameters()))]
        fc_params = model.model.last_linear.parameters()
        to_optim = [{'params': all_but_fc_params, 'lr': opt.lr, 'weight_decay': opt.decay},
                    {'params': fc_params, 'lr': opt.fc_lr, 'weight_decay': opt.decay}]

    model.to(opt.device)

    # DATALOADER SETUPS
    dataloaders, train_data_sampler = get_dataloaders(opt, model)

    opt.n_classes = len(dataloaders['training'].dataset.avail_classes)

    # CREATE LOGGING FILES
    sub_loggers = ['Train', 'Test', 'Model Grad']
    if opt.use_tv_split:
        sub_loggers.append('Val')
    LOG = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True)

    # LOSS SETUP
    embed_dim = opt.embed_dim
    sampling = bmine.select(opt.batch_mining, opt)
    criterion, _, loss_lib = criteria.select(opt.loss, opt, sampling)

    if loss_lib.REQUIRES_OPTIM:
        if hasattr(criterion, 'optim_dict_list') and criterion.optim_dict_list is not None:
            to_optim += criterion.optim_dict_list
        else:
            to_optim += [{'params': criterion.parameters(), 'lr': criterion.lr}]

    criterion.to(opt.device)

    if train_data_sampler and 'criterion' in train_data_sampler.name:
        train_data_sampler.internal_criterion = criterion

    embedding_producer = None
    if opt.embedding_producer_used:
        embedding_producer = EmbeddingProducer(
            num_classes=opt.n_classes, dim=opt.embed_dim, produce_type=opt.produce_type,
            # for das
            das_num_produce=opt.das_num_produce, das_normalize=not opt.das_not_normalize,
            # for dfs
            das_dfs_num_scale=opt.das_dfs_num_scale,
            das_dfs_scale_left=opt.das_dfs_scale_left,
            das_dfs_scale_right=opt.das_dfs_scale_right,
            # for mts
            das_mts_num_transformation_bank=opt.das_mts_num_transformation_bank,
            das_mts_scale=opt.das_mts_scale,
            # others
            aux_loss=opt.aux_loss,
            aux_loss_weight=opt.aux_loss_weight,
            detach=opt.detach,
        )
        print("[EmbeddingProducer] Using EmbeddingProducer for model training")
        print(embedding_producer)

    sec = None
    if opt.sec:
        sec = SEC(momentum=opt.sec_momentum, weight=opt.sec_wei)

    # OPTIM SETUP
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(to_optim)
    elif opt.optim == 'sgd':
        optimizer = torch.optim.SGD(to_optim, momentum=0.9)
    elif opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(to_optim)
    else:
        raise Exception('Optimizer <{}> not available!'.format(opt.optim))
    if opt.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)
    elif opt.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_tau, gamma=opt.gamma)
    else:
        raise Exception('Scheduler <{}> not available!'.format(opt.scheduler))

    # METRIC COMPUTER
    opt.rho_spectrum_embed_dim = opt.embed_dim
    # automatically switch R@1,2,4,8 for CUB, CARS into
    # R@1,10,100,1000 for SOP and In-Shop
    if opt.dataset == 'online_products' or opt.dataset == 'in-shop':
        for ori, tar in zip([2, 4, 8], [10, 100, 1000]):
            original_recall = 'e_recall@{}'.format(ori)
            target_recall = 'e_recall@{}'.format(tar)
            if original_recall in opt.evaluation_metrics:
                opt.evaluation_metrics.remove(original_recall)
                opt.evaluation_metrics.append(target_recall)

    metric_computer = metric.MetricComputer(opt.evaluation_metrics, opt)

    # Summary
    data_text = 'Dataset:\t {}'.format(opt.dataset.upper())
    setup_text = 'Objective:\t {}'.format(opt.loss.upper())
    miner_text = 'SAMPLING:\t {}'.format(opt.batch_mining if criterion.REQUIRES_SAMPLING else 'N/A')
    arch_text = 'Backbone:\t {} (#weights: {})'.format(opt.arch.upper(), misc.gimme_params(model))
    summary = data_text + '\n' + setup_text + '\n' + miner_text + '\n' + arch_text
    print(summary)

    for epoch in range(opt.n_epochs):
        epoch_start_time = time.time()

        if epoch > 0 and opt.data_idx_full_prec and train_data_sampler and train_data_sampler.requires_storage:
            train_data_sampler.full_storage_update(dataloaders['evaluation'], model, opt.device)

        train_one_epoch(opt, epoch, scheduler, train_data_sampler, dataloaders['training'],
                        model, criterion, optimizer, LOG, embedding_producer=embedding_producer, sec=sec)
        evaluate(opt, epoch, model, dataloaders, metric_computer, LOG, criterion=criterion)

        print('Total Epoch Runtime: {0:4.2f}s'.format(time.time() - epoch_start_time))

    # CREATE A SUMMARY TEXT FILE
    summary_text = ''
    full_training_time = time.time() - full_training_start_time
    summary_text += 'Training Time: {} min.\n'.format(np.round(full_training_time / 60, 2))

    for sub_logger in LOG.sub_loggers:
        metrics = LOG.graph_writer[sub_logger].ov_title
        summary_text += '{} metric: {}\n'.format(sub_logger.upper(), metrics)

    with open(LOG.save_path + '/training_summary.txt', 'w') as summary_file:
        summary_file.write(summary_text)

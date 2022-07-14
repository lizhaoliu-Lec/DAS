import os
import random
import time

import numpy as np
import seaborn as sns
import torch.multiprocessing
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasampler as dsamplers
import dataset
import evaluation as eval
from modules.embedding_producer import EmbeddingProducer
from modules.sec import SEC


def set_seed(seed):
    """
    set seed for reproducibility
    """
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(opt, model):
    dataloaders = {}
    datasets = dataset.select(opt.dataset, opt, opt.source_path)

    dataloaders['evaluation'] = DataLoader(datasets['evaluation'], num_workers=opt.kernels,
                                           batch_size=opt.bs, shuffle=False)
    if opt.dataset == "in-shop":
        dataloaders['testing_query'] = DataLoader(datasets['testing'], num_workers=opt.kernels,
                                                  batch_size=opt.bs,
                                                  shuffle=False)
        dataloaders['testing_gallery'] = DataLoader(datasets['evaluation_train'], num_workers=opt.kernels,
                                                    batch_size=opt.bs,
                                                    shuffle=False)
    else:
        dataloaders['testing'] = DataLoader(datasets['testing'], num_workers=opt.kernels,
                                            batch_size=opt.bs,
                                            shuffle=False)

    if opt.use_tv_split:
        dataloaders['validation'] = DataLoader(datasets['validation'], num_workers=opt.kernels,
                                               batch_size=opt.bs, shuffle=False)

    if hasattr(datasets['training'], 'class_ix2super_ix'):
        opt.class_ix2super_ix = datasets['training'].class_ix2super_ix

    train_data_sampler = dsamplers.select(opt.data_sampler, opt, datasets['training'].image_dict,
                                          datasets['training'].image_list)
    if train_data_sampler is not None:

        if train_data_sampler.requires_storage:
            train_data_sampler.create_storage(dataloaders['evaluation'], model, opt.device)

        dataloaders['training'] = DataLoader(datasets['training'], num_workers=opt.kernels,
                                             batch_sampler=train_data_sampler)
    else:
        dataloaders['training'] = DataLoader(datasets['training'], num_workers=opt.kernels,
                                             batch_size=opt.bs,
                                             shuffle=True, drop_last=True, pin_memory=True)

    return dataloaders, train_data_sampler


def train_one_epoch(opt, epoch, scheduler, train_data_sampler, dataloader, model, criterion, optimizer, LOG,
                    embedding_producer: EmbeddingProducer = None, sec: SEC = None):
    opt.epoch = epoch
    # Scheduling Changes specifically for cosine scheduling
    if opt.scheduler != 'none':
        print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

    if train_data_sampler and train_data_sampler.requires_storage:
        train_data_sampler.precompute_indices()

    # Train one epoch
    start = time.time()
    model.train()

    loss_collect = []
    data_iterator = tqdm(dataloader, desc='Epoch {}/{} Training...'.format(epoch, opt.n_epochs))
    if opt.n_warm > 0 and epoch < opt.n_warm:
        print('Using warmup strategy for epoch {}'.format(epoch))
    loss_args = {'batch': None, 'labels': None, 'batch_features': None, 'f_embed': None}

    for i, out in enumerate(data_iterator):
        global_steps = epoch * len(data_iterator) + i
        opt.iteration = global_steps

        class_labels, input, input_indices, super_class_labels = out

        # Compute Embedding
        input = input.to(opt.device)
        model_args = {'x': input.to(opt.device)}
        # Needed for MixManifold settings.
        if 'mix' in opt.arch:
            model_args['labels'] = class_labels
        if opt.n_warm > 0 and epoch < opt.n_warm:
            model_args['warmup'] = True
        embeds = model(**model_args)
        features = None
        if isinstance(embeds, tuple):
            embeds, (avg_features, features) = embeds

        if embedding_producer is not None:
            embeds, class_labels = embedding_producer(embeds, class_labels)
            LOG.tensorboard.add_scalar(tag='EmbeddingProducer/Distance',
                                       scalar_value=embedding_producer.embedding_distance,
                                       global_step=global_steps)

        # Compute Loss
        if opt.embedding_producer_used and opt.aux_loss:
            # main loss
            loss_args['batch_features'] = features
            loss_args['batch'] = embeds[:opt.bs, ...]
            loss_args['labels'] = class_labels[:opt.bs, ...]
            loss_args['f_embed'] = model.model.last_linear
            loss_args['tensorboard'] = LOG.tensorboard
            loss = criterion(**loss_args)
            # aux loss
            loss_args['batch'] = embeds
            loss_args['labels'] = class_labels
            aux_loss = criterion(**loss_args)
            # log aux loss to tensorboard
            LOG.tensorboard.add_scalar(tag='EmbeddingProducer/AuxLoss',
                                       scalar_value=aux_loss.item(),
                                       global_step=global_steps)

            # all loss
            loss = loss + opt.aux_loss_weight * aux_loss
        else:
            loss_args['batch_features'] = features
            loss_args['batch'] = embeds
            loss_args['labels'] = class_labels
            loss_args['f_embed'] = model.model.last_linear
            loss_args['tensorboard'] = LOG.tensorboard
            loss = criterion(**loss_args)

        # plus sec loss here
        if sec is not None:
            sec_loss = sec(model.x_before_normed)
            LOG.tensorboard.add_scalar(tag='SEC/Loss',
                                       scalar_value=sec_loss.item(),
                                       global_step=opt.epoch)
            LOG.tensorboard.add_scalar(tag='SEC/NormMean',
                                       scalar_value=sec.norm_mean.item(),
                                       global_step=opt.epoch)
            loss += sec.weight * sec_loss

        optimizer.zero_grad()
        loss.backward()

        # Compute Model Gradients and log them!
        grads = np.concatenate(
            [p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
        grad_l2, grad_max = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
        LOG.progress_saver['Model Grad'].log('Grad L2', grad_l2, group='L2')
        LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')
        LOG.tensorboard.add_scalar(tag='Grad/L2', scalar_value=grad_l2, global_step=global_steps)
        LOG.tensorboard.add_scalar(tag='Grad/Max', scalar_value=grad_max, global_step=global_steps)

        # Update network weights!
        optimizer.step()

        loss_collect.append(loss.item())

        if i == len(dataloader) - 1:
            data_iterator.set_description(
                'Epoch (Train) {0}/{1}: Mean Loss [{2:.4f}]'.format(epoch, opt.n_epochs, np.mean(loss_collect)))

        # A brilliant way to update embeddings!
        if train_data_sampler and train_data_sampler.requires_storage and train_data_sampler.update_storage:
            train_data_sampler.replace_storage_entries(embeds.detach().cpu(), input_indices)

    result_metrics = {'loss': np.mean(loss_collect)}

    LOG.progress_saver['Train'].log('epochs', epoch)
    for metric_name, metric_val in result_metrics.items():
        LOG.progress_saver['Train'].log(metric_name, metric_val)
        LOG.tensorboard.add_scalar(tag='Train/%s' % metric_name, scalar_value=metric_val, global_step=epoch)
    LOG.progress_saver['Train'].log('time', np.round(time.time() - start, 4))
    LOG.tensorboard.add_scalar(tag='Train/time', scalar_value=np.round(time.time() - start, 4), global_step=epoch)

    # Learning Rate Scheduling Step
    if opt.scheduler != 'none':
        scheduler.step()


@torch.no_grad()
def evaluate(opt, epoch, model, dataloaders, metric_computer, LOG, criterion=None):
    # Evaluate Metric for Training & Test (& Validation)
    model.eval()
    print('\nEpoch {0}/{1} Computing Testing Metrics...'.format(epoch, opt.n_epochs))
    if opt.dataset == "in-shop":
        eval.evaluate_query_and_gallery(LOG, metric_computer, dataloaders['testing_query'],
                                        dataloaders['testing_gallery'], model, opt, opt.eval_types,
                                        opt.device, log_key='Test', criterion=criterion)
    else:
        eval.evaluate(LOG, metric_computer, dataloaders['testing'], model, opt, opt.eval_types,
                      opt.device, log_key='Test', criterion=criterion)
    if opt.use_tv_split:
        print('\nEpoch {0}/{1} Computing Validation Metrics...'.format(epoch, opt.n_epochs))
        eval.evaluate(LOG, metric_computer, dataloaders['validation'], model, opt, opt.eval_types,
                      opt.device, log_key='Val', criterion=criterion)
    if not opt.no_train_metrics:  # only perform evaluation on training set if no_train_metrics is not specified
        print('\nEpoch {0}/{1} Computing Training Metrics...'.format(epoch, opt.n_epochs))
        eval.evaluate(LOG, metric_computer, dataloaders['evaluation'], model, opt, opt.eval_types,
                      opt.device, log_key='Train', criterion=criterion)

    LOG.update(update_all=True)


def normalize_image(x):
    x = np.transpose(x, (1, 2, 0))
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return (x * 255).astype(np.uint8)

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from utilities.logger import LOGGER


def evaluate(LOG: LOGGER, metric_computer, dataloader, model, opt, eval_types, device,
             aux_store=None, make_recall_plot=False, log_key='Test', criterion=None):
    """
    Parent-Function to compute evaluation metric, print summary string and
    store checkpoint files/plot sample recall plots.
    """
    computed_metrics, extra_infos = metric_computer.compute_standard(opt, model, dataloader, eval_types, device)

    numeric_metrics = {}
    histogram_metrics = {}
    for main_key in computed_metrics.keys():
        for name, value in computed_metrics[main_key].items():
            if isinstance(value, np.ndarray):
                if main_key not in histogram_metrics:
                    histogram_metrics[main_key] = {}
                histogram_metrics[main_key][name] = value
            else:
                if main_key not in numeric_metrics:
                    numeric_metrics[main_key] = {}
                numeric_metrics[main_key][name] = value

    full_result_str = ''
    for eval_type in numeric_metrics.keys():
        full_result_str += 'Embed-Type: {}:\n'.format(eval_type)
        for i, (metric_name, metric_val) in enumerate(numeric_metrics[eval_type].items()):
            full_result_str += '{0}{1}: {2:4.4f}'.format(' | ' if i > 0 else '', metric_name, metric_val)
        full_result_str += '\n'

    print(full_result_str)

    for eval_type in eval_types:
        for storage_metric in opt.storage_metrics:
            parent_metric = eval_type + '_{}'.format(storage_metric.split('@')[0])
            if parent_metric not in LOG.progress_saver[log_key].groups.keys() or \
                    numeric_metrics[eval_type][storage_metric] > np.max(
                LOG.progress_saver[log_key].groups[parent_metric][storage_metric]['content']):
                print('Saved weights for best {}: {}\n'.format(log_key, parent_metric))
                set_checkpoint(model, opt, LOG.progress_saver,
                               LOG.save_path + '/checkpoint_{}_{}_{}.pth.tar'.format(log_key, eval_type,
                                                                                     storage_metric),
                               aux=aux_store)
                if criterion is not None:
                    set_checkpoint(criterion, opt, LOG.progress_saver,
                                   LOG.save_path + '/criterion_checkpoint_{}_{}_{}.pth.tar'.format(log_key, eval_type,
                                                                                                   storage_metric),
                                   aux=aux_store)

    # log histogram to tensorboard
    for eval_type in histogram_metrics.keys():
        for eval_metric, hist in histogram_metrics[eval_type].items():
            LOG.tensorboard.add_histogram(tag='%s/%s' % (log_key, eval_type + '_%s' % eval_metric),
                                          values=hist, global_step=opt.epoch)
            LOG.tensorboard.add_histogram(tag='%s/%s' % (log_key, eval_type + '_LOG-%s' % eval_metric),
                                          values=np.log(hist) + 20, global_step=opt.epoch)

    for eval_type in numeric_metrics.keys():
        for eval_metric in numeric_metrics[eval_type].keys():
            parent_metric = eval_type + '_{}'.format(eval_metric.split('@')[0])
            LOG.progress_saver[log_key].log(eval_metric, numeric_metrics[eval_type][eval_metric], group=parent_metric)
            LOG.tensorboard.add_scalar(tag='%s/%s' % (log_key, eval_type + '_%s' % eval_metric),
                                       scalar_value=numeric_metrics[eval_type][eval_metric], global_step=opt.epoch)

        if make_recall_plot:
            recover_closest_standard(extra_infos[eval_type]['features'],
                                     extra_infos[eval_type]['image_paths'],
                                     LOG.prop.save_path + '/sample_recoveries.png')


def evaluate_query_and_gallery(LOG: LOGGER, metric_computer, query, gallery, model, opt, eval_types, device,
                               aux_store=None, make_recall_plot=False, log_key='Test', criterion=None):
    """
    Parent-Function to compute evaluation metric, print summary string and
    store checkpoint files/plot sample recall plots.
    """
    computed_metrics, extra_infos = metric_computer.compute_query_gallery(opt, model, query, gallery, eval_types,
                                                                          device)

    numeric_metrics = {}
    histogram_metrics = {}
    for main_key in computed_metrics.keys():
        for name, value in computed_metrics[main_key].items():
            if isinstance(value, np.ndarray):
                if main_key not in histogram_metrics:
                    histogram_metrics[main_key] = {}
                histogram_metrics[main_key][name] = value
            else:
                if main_key not in numeric_metrics:
                    numeric_metrics[main_key] = {}
                numeric_metrics[main_key][name] = value

    full_result_str = ''
    for eval_type in numeric_metrics.keys():
        full_result_str += 'Embed-Type: {}:\n'.format(eval_type)
        for i, (metric_name, metric_val) in enumerate(numeric_metrics[eval_type].items()):
            full_result_str += '{0}{1}: {2:4.4f}'.format(' | ' if i > 0 else '', metric_name, metric_val)
        full_result_str += '\n'

    print(full_result_str)

    for eval_type in eval_types:
        for storage_metric in opt.storage_metrics:
            parent_metric = eval_type + '_{}'.format(storage_metric.split('@')[0])
            if parent_metric not in LOG.progress_saver[log_key].groups.keys() or \
                    numeric_metrics[eval_type][storage_metric] > np.max(
                LOG.progress_saver[log_key].groups[parent_metric][storage_metric]['content']):
                print('Saved weights for best {}: {}\n'.format(log_key, parent_metric))
                set_checkpoint(model, opt, LOG.progress_saver,
                               LOG.save_path + '/checkpoint_{}_{}_{}.pth.tar'.format(log_key, eval_type,
                                                                                     storage_metric),
                               aux=aux_store)
                if criterion is not None:
                    set_checkpoint(criterion, opt, LOG.progress_saver,
                                   LOG.save_path + '/criterion_checkpoint_{}_{}_{}.pth.tar'.format(log_key, eval_type,
                                                                                                   storage_metric),
                                   aux=aux_store)

    if opt.log_online:
        for eval_type in histogram_metrics.keys():
            for eval_metric, hist in histogram_metrics[eval_type].items():
                import wandb
                wandb.log({log_key + ': ' + eval_type + '_{}'.format(eval_metric): wandb.Histogram(
                    np_histogram=(list(hist), list(np.arange(len(hist) + 1))))}, step=opt.epoch)
                wandb.log({log_key + ': ' + eval_type + '_LOG-{}'.format(eval_metric): wandb.Histogram(
                    np_histogram=(list(np.log(hist) + 20), list(np.arange(len(hist) + 1))))}, step=opt.epoch)

    # log histogram to tensorboard
    for eval_type in histogram_metrics.keys():
        for eval_metric, hist in histogram_metrics[eval_type].items():
            LOG.tensorboard.add_histogram(tag='%s/%s' % (log_key, eval_type + '_%s' % eval_metric),
                                          values=hist, global_step=opt.epoch)
            LOG.tensorboard.add_histogram(tag='%s/%s' % (log_key, eval_type + '_LOG-%s' % eval_metric),
                                          values=np.log(hist) + 20, global_step=opt.epoch)

    for eval_type in numeric_metrics.keys():
        for eval_metric in numeric_metrics[eval_type].keys():
            parent_metric = eval_type + '_{}'.format(eval_metric.split('@')[0])
            LOG.progress_saver[log_key].log(eval_metric, numeric_metrics[eval_type][eval_metric], group=parent_metric)
            LOG.tensorboard.add_scalar(tag='%s/%s' % (log_key, eval_type + '_%s' % eval_metric),
                                       scalar_value=numeric_metrics[eval_type][eval_metric], global_step=opt.epoch)

        if make_recall_plot:
            recover_closest_standard(extra_infos[eval_type]['features'],
                                     extra_infos[eval_type]['image_paths'],
                                     LOG.prop.save_path + '/sample_recoveries.png')


def set_checkpoint(model, opt, progress_saver, save_path, aux=None):
    if 'experiment' in vars(opt):
        import argparse
        save_opt = {key: item for key, item in vars(opt).items() if key != 'experiment'}
        save_opt = argparse.Namespace(**save_opt)
    else:
        save_opt = opt

    torch.save({'state_dict': model.state_dict(), 'opt': save_opt, 'progress': progress_saver, 'aux': aux}, save_path)


def recover_closest_standard(feature_matrix_all, image_paths, save_path, n_image_samples=10, n_closest=3):
    image_paths = np.array([x[0] for x in image_paths])
    sample_idxs = np.random.choice(np.arange(len(feature_matrix_all)), n_image_samples)

    faiss_search_index = faiss.IndexFlatL2(feature_matrix_all.shape[-1])
    faiss_search_index.add(feature_matrix_all)
    _, closest_feature_idxs = faiss_search_index.search(feature_matrix_all, n_closest + 1)

    sample_paths = image_paths[closest_feature_idxs][sample_idxs]

    f, axes = plt.subplots(n_image_samples, n_closest + 1)
    for i, (ax, plot_path) in enumerate(zip(axes.reshape(-1), sample_paths.reshape(-1))):
        ax.imshow(np.array(Image.open(plot_path)))
        ax.set_xticks([])
        ax.set_yticks([])
        if i % (n_closest + 1):
            ax.axvline(x=0, color='g', linewidth=13)
        else:
            ax.axvline(x=0, color='r', linewidth=13)
    f.set_size_inches(10, 20)
    f.tight_layout()
    f.savefig(save_path)
    plt.close()

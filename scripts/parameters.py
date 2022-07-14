import argparse
import os


def get_basic_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training_script', default='standard', type=str, help='Script used to train the model.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducibility.')
    return parser


def get_basic_training_parameters(parser):
    # Dataset-related Parameters
    parser.add_argument('--dataset', default='cub200', type=str,
                        choices=['cub200', 'cars196', 'online_products', 'in-shop'],
                        help='Dataset to use. Currently supported: cub200, cars196, online_products, in_shop.')
    parser.add_argument('--use_tv_split', action='store_true',
                        help='Flag. If set, split the training set into a training/validation set.')
    parser.add_argument('--tv_split_by_samples', action='store_true',
                        help='Flag. If set, create the validation set by taking a percentage of samples PER class. '
                             'Otherwise, the validation set is create by taking a percentage of classes.')
    parser.add_argument('--tv_split_perc', default=0.8, type=float,
                        help='Percentage with which the training dataset is split into training/validation.')
    parser.add_argument('--augmentation', default='base', type=str, choices=['base', 'adv', 'big', 'red'],
                        help='Type of preprocessing/augmentation to use on the data. '
                             'Available: base (standard), adv (with color/brightness changes), '
                             'big (Images of size 256x256), red (No RandomResizedCrop).')

    # General Training Parameters
    parser.add_argument('--lr', default=0.00001, type=float, help='Learning Rate for network parameters.')
    parser.add_argument('--drop', default=-1, type=float, help='Dropout rate on the pooled features.')
    parser.add_argument('--fc_lr', default=-1, type=float,
                        help='Optional. If not -1, sets the learning rate for the final linear embedding layer.')
    parser.add_argument('--decay', default=0.0004, type=float, help='Weight decay placed on network weights.')
    parser.add_argument('--n_epochs', default=150, type=int, help='Number of training epochs.')
    parser.add_argument('--n_warm', default=0, type=int, help='Number of epochs to warmup.')
    parser.add_argument('--kernels', default=6, type=int, help='Number of workers for pytorch dataloader.')
    parser.add_argument('--bs', default=112, type=int, help='Mini-Batchsize to use.')
    parser.add_argument('--scheduler', default='multistep', type=str, choices=['multistep', 'step'],
                        help='Type of learning rate scheduling. Currently supported: multistep & step')
    parser.add_argument('--gamma', default=0.3, type=float, help='Learning rate reduction after tau epochs.')
    parser.add_argument('--tau', default=[1000], nargs='+', type=int, help='Stepsize before reducing learning rate'
                                                                           'for multistepLR scheduler.')
    parser.add_argument('--step_tau', default=10, type=int, help='Stepsize before reducing learning rate'
                                                                 'for stepLR scheduler.')

    # Loss-specific Settings
    parser.add_argument('--optim', default='adam', type=str, choices=['adam', 'sgd', 'adamw'],
                        help='Optimization method to use. Currently supported: adam & sgd & adamw.')
    parser.add_argument('--loss', default='softmax', type=str,
                        help='Training loss: For supported methods, please check loss/__init__.py')
    parser.add_argument('--batch_mining', default='distance', type=str,
                        help='SAMPLING for tuple-based losses: '
                             'For supported methods, please check batch_mining/__init__.py')

    # Network-related Flags
    parser.add_argument('--embed_dim', default=128, type=int,
                        help='Embedding dimensionality of the network. '
                             'Note: dim = 64, 128 or 512 is used in most papers, depending on the architecture.')
    parser.add_argument('--not_pretrained', action='store_true',
                        help='Flag. If set, no ImageNet pretraining is used to initialize the network.')
    parser.add_argument('--arch', default='resnet50_frozen_normalize', type=str,
                        help='Underlying network architecture. '
                             'Frozen denotes that existing pretrained batchnorm layers are frozen, and '
                             'normalize denotes normalization of the output embedding.')
    parser.add_argument('--ckpt_path', default=None, type=str,
                        help='The path to load pretrained model checkpoint.')

    # Evaluation Parameters
    parser.add_argument('--no_train_metrics', action='store_true',
                        help='Flag. If set, evaluation metric are not computed for the training data. '
                             'Saves a forward pass over the full training dataset.')
    parser.add_argument('--evaluate_on_gpu', action='store_true',
                        help='Flag. If set, all metric, when possible, are computed on the GPU (requires Faiss-GPU).')
    parser.add_argument('--evaluation_metrics', nargs='+',
                        default=['e_recall@1', 'e_recall@2', 'e_recall@4', 'e_recall@8', 'nmi', 'f1', 'mAP_1000',
                                 'mAP_lim', 'mAP_c',
                                 'dists@intra', 'dists@inter', 'dists@intra_over_inter', 'rho_spectrum@0',
                                 'rho_spectrum@-1', 'rho_spectrum@1', 'rho_spectrum@2', 'rho_spectrum@10'],
                        type=str, help='Metrics to evaluate performance by.')

    parser.add_argument('--storage_metrics', nargs='+', default=['e_recall@1'], type=str,
                        help='Improvement in these metric on a dataset trigger checkpointing.')
    parser.add_argument('--eval_types', nargs='+', default=['discriminative'], type=str,
                        help='The network may produce multiple embeddings (ModuleDict, relevant for e.g. DiVA). '
                             'If the key is listed here, the entry will be evaluated on the evaluation metric. '
                             'Note: One may use Combined [embed1, embed2, ..., embedn], '
                             '[w1, w2, ..., wn] to compute evaluation metric on weighted (normalized) combinations.')

    # Setup Parameters
    parser.add_argument('--gpu', default=[0], nargs='+', type=int, help='Gpu to use.')
    parser.add_argument('--savename', default='group_plus_seed', type=str,
                        help='Run savename - if default, '
                             'the savename will comprise the project and group name (see wandb_parameters()).')
    parser.add_argument('--source_path', default='<folder_path_that_stores_cub200_cars196_online_products>.', type=str,
                        help='Path to training data.')
    parser.add_argument('--save_path', default=os.getcwd() + '/TrainingResults', type=str,
                        help='Where to save everything.')

    return parser


def get_loss_specific_parameters(parser):
    # Contrastive Loss
    parser.add_argument('--loss_contrastive_pos_margin', default=0, type=float,
                        help='positive margin for contrastive pairs.')
    parser.add_argument('--loss_contrastive_neg_margin', default=1, type=float,
                        help='negative margin for contrastive pairs.')
    parser.add_argument('--loss_contrastive_stop_pos', action='store_true',
                        help='Whether to stop the gradients of the positive during the loss computation.')
    parser.add_argument('--loss_contrastive_stop_neg', action='store_true',
                        help='Whether to stop the gradients of the negative during the loss computation.')

    # Triplet-based Losses
    parser.add_argument('--loss_triplet_margin', default=0.2, type=float, help='Margin for Triplet Loss')

    # MarginLoss
    parser.add_argument('--loss_margin_margin', default=0.2, type=float, help='Triplet margin.')
    parser.add_argument('--loss_margin_beta_lr', default=0.0005, type=float,
                        help='Learning Rate for learnable class margin parameters in MarginLoss')
    parser.add_argument('--loss_margin_beta', default=1.2, type=float,
                        help='Initial Class Margin Parameter in Margin Loss')
    parser.add_argument('--loss_margin_nu', default=0, type=float,
                        help='Regularisation value on betas in Margin Loss. Generally not needed.')
    parser.add_argument('--loss_margin_beta_constant', action='store_true',
                        help='Flag. If set, beta-values are left untrained.')

    # ProxyNCA
    parser.add_argument('--loss_proxynca_lrmulti', default=50, type=float,
                        help='Learning Rate multiplier for Proxies in proxynca. '
                             'NOTE: The number of proxies is determined by the number of data classes')

    # ProxyAnchor
    parser.add_argument('--loss_proxyanchor_margin', default=0.1, type=float,
                        help='Margin in proxyanchor loss.')
    parser.add_argument('--loss_proxyanchor_alpha', default=32, type=float,
                        help='Alpha in proxyanchor loss.')
    parser.add_argument('--loss_proxyanchor_lrmulti', default=100, type=float,
                        help='Learning Rate multiplier for Proxies in proxyanchor. '
                             'NOTE: The number of proxies is determined by the number of data classes')
    parser.add_argument('--loss_proxyanchor_decay', default=0.0, type=float,
                        help='Decay for proxies in proxyanchor loss.')

    # NPair
    parser.add_argument('--loss_npair_l2', default=0.005, type=float,
                        help='L2 weight in NPair. '
                             'Note: Set to 0.02 in paper, but multiplied with 0.25 in their implementation.')

    # Angular Loss
    parser.add_argument('--loss_angular_alpha', default=45, type=float, help='Angular margin in degrees.')
    parser.add_argument('--loss_angular_npair_ang_weight', default=2, type=float,
                        help='Relative weighting between angular and npair contribution.')
    parser.add_argument('--loss_angular_npair_l2', default=0.005, type=float,
                        help='L2 weight on NPair (as embeddings are not normalized).')

    # Multisimilary Loss
    parser.add_argument('--loss_multisimilarity_pos_weight', default=2, type=float,
                        help='Weighting on positive similarities.')
    parser.add_argument('--loss_multisimilarity_neg_weight', default=40, type=float,
                        help='Weighting on negative similarities.')
    parser.add_argument('--loss_multisimilarity_margin', default=0.1, type=float,
                        help='Distance margin for both positive and negative similarities.')
    parser.add_argument('--loss_multisimilarity_thresh', default=0.5, type=float, help='Exponential thresholding.')

    # Lifted Structure Loss
    parser.add_argument('--loss_lifted_neg_margin', default=1, type=float, help='Margin placed on similarities.')
    parser.add_argument('--loss_lifted_l2', default=0.005, type=float,
                        help='As embeddings are not normalized, they need to be placed under penalty.')

    # Quadruplet Loss
    parser.add_argument('--loss_quadruplet_margin_alpha_1', default=0.2, type=float,
                        help='Quadruplet Loss requires two margins. This is the first one.')
    parser.add_argument('--loss_quadruplet_margin_alpha_2', default=0.2, type=float, help='This is the second.')

    # Soft-Triple Loss
    parser.add_argument('--loss_softtriplet_n_centroids', default=2, type=int, help='Number of proxies per class.')
    parser.add_argument('--loss_softtriplet_margin_delta', default=0.01, type=float,
                        help='Margin placed on sample-proxy similarities.')
    parser.add_argument('--loss_softtriplet_gamma', default=0.1, type=float,
                        help='Weight over sample-proxies within a class.')
    parser.add_argument('--loss_softtriplet_lambda', default=8, type=float, help='Serves as a temperature.')
    parser.add_argument('--loss_softtriplet_reg_weight', default=0.2, type=float,
                        help='Regularization weight on the number of proxies.')
    parser.add_argument('--loss_softtriplet_lrmulti', default=1, type=float,
                        help='Learning Rate multiplier for proxies.')

    # Normalized Softmax Loss
    parser.add_argument('--loss_softmax_lr', default=0.00001, type=float, help='Learning rate on class proxies.')
    parser.add_argument('--loss_softmax_temperature', default=0.05, type=float, help='Temperature for NCA objective.')

    # Histogram Loss
    parser.add_argument('--loss_histogram_nbins', default=65, type=int,
                        help='Number of bins for histogram discretization.')

    # SNR Triplet (with learnable margin) Loss
    parser.add_argument('--loss_snr_margin', default=0.2, type=float, help='Triplet margin.')
    parser.add_argument('--loss_snr_reg_lambda', default=0.005, type=float,
                        help='Regularization of in-batch element sum.')

    # ArcFace
    parser.add_argument('--loss_arcface_lr', default=0.0005, type=float, help='Learning rate on class proxies.')
    parser.add_argument('--loss_arcface_angular_margin', default=0.5, type=float, help='Angular margin in radians.')
    parser.add_argument('--loss_arcface_feature_scale', default=16, type=float,
                        help='Inverse Temperature for NCA objective.')
    return parser


def get_sampling_specific_parameters(parser):
    # Distance-based SAMPLING
    parser.add_argument('--miner_distance_lower_cutoff', default=0.5, type=float,
                        help='Lower cutoff on distances - values below are sampled with equal prob.')
    parser.add_argument('--miner_distance_upper_cutoff', default=1.4, type=float,
                        help='Upper cutoff on distances - values above are IGNORED.')
    # Spectrum-Regularized Miner (as proposed in our paper) - utilizes a distance-based sampler that is regularized.
    parser.add_argument('--miner_rho_distance_lower_cutoff', default=0.5, type=float,
                        help='Lower cutoff on distances - values below are sampled with equal prob.')
    parser.add_argument('--miner_rho_distance_upper_cutoff', default=1.4, type=float,
                        help='Upper cutoff on distances - values above are IGNORED.')
    parser.add_argument('--miner_rho_distance_cp', default=0.2, type=float,
                        help='Probability to replace a negative with a positive.')
    return parser


def get_batch_creation_parameters(parser):
    parser.add_argument('--data_sampler', default='class_random', type=str,
                        help='How the batch is created. Available options: See datasampler/__init__.py.')
    parser.add_argument('--samples_per_class', default=2, type=int,
                        help='Number of samples in one class drawn before choosing the next class. '
                             'Set to >1 for tuple-based loss.')
    # Batch-Sample Flags - Have no relevance to default SPC-N sampling
    parser.add_argument('--data_batchmatch_bigbs', default=512, type=int,
                        help='Size of batch to be summarized into a smaller batch. '
                             'For distillation/coreset-based methods.')
    parser.add_argument('--data_batchmatch_ncomps', default=10, type=int,
                        help='Number of batch candidates that are evaluated, from which the best one is chosen.')
    parser.add_argument('--data_storage_no_update', action='store_true',
                        help='Flag for methods that need a sample storage. If set, storage entries are NOT updated.')
    parser.add_argument('--data_d2_coreset_lambda', default=1, type=float, help='Regularisation for D2-coreset.')
    parser.add_argument('--data_gc_coreset_lim', default=1e-9, type=float, help='D2-coreset value limit.')
    parser.add_argument('--data_sampler_lowproj_dim', default=-1, type=int,  # TODO look into the related paper
                        help='Optionally project embeddings into a lower dimension to ensure that '
                             'greedy coreset works better. Only makes a difference for large embedding dims.')
    parser.add_argument('--data_sim_measure', default='euclidean', type=str,
                        help='Distance measure to use for batch selection.')
    parser.add_argument('--data_gc_softened', action='store_true',
                        help='Flag. If set, use a soft version of greedy coreset.')
    parser.add_argument('--data_idx_full_prec', action='store_true', help='Deprecated.')
    parser.add_argument('--data_mb_mom', default=-1, type=float,
                        help='For memory-bank based samplers - momentum term on storage entry updates.')
    parser.add_argument('--data_mb_lr', default=1, type=float, help='Deprecated.')
    parser.add_argument('--data_num_super', default=1, type=int, help='Number of super class to sample for each batch.')

    return parser


def get_embedding_producer_parameters(parser):
    parser.add_argument('--embedding_producer_used', action='store_true',
                        help='Flag. If set, embedding producer is used to train the network.')
    parser.add_argument('--produce_type', default='embeddings', type=str,
                        help='The type to produce embeddings. Currently only support ["DAS"]')

    # for das
    parser.add_argument('--das_num_produce', default=3, type=int,
                        help='The number of embedding to produce in das.')
    parser.add_argument('--das_not_normalize', action='store_true',
                        help='If set, do not normalize the produced embeddings.')

    # for dfs
    parser.add_argument('--das_dfs_num_scale', default=4, type=int,
                        help='The number of neural to scale in dfs.')
    parser.add_argument('--das_dfs_scale_left', default=0.5, type=float,
                        help='The left range to scale the neurons.')
    parser.add_argument('--das_dfs_scale_right', default=2.0, type=float,
                        help='The right range to scale the neurons.')

    # for mts
    parser.add_argument('--das_mts_num_transformation_bank', default=10, type=int,
                        help='The bank capacity for intra-class transformation.')
    parser.add_argument('--das_mts_scale', default=0.01, type=float,
                        help='The scale to shift the embedding.')

    # for aux loss
    parser.add_argument('--aux_loss', action='store_true',
                        help='Flag. If set, apply the normal loss to real embeddings first, '
                             'then apply aux loss to both real and produced embeddings.')
    parser.add_argument('--aux_loss_weight', default=1.0, type=float,
                        help='The weight for aux loss.')
    parser.add_argument('--detach', action='store_true',
                        help='Flag. If set, apply the detach to the produced embedding.')

    return parser


def read_arguments_from_cmd():
    parser = get_basic_parameters()
    parser = get_basic_training_parameters(parser)
    parser = get_batch_creation_parameters(parser)
    parser = get_sampling_specific_parameters(parser)
    parser = get_loss_specific_parameters(parser)
    # for embedding producer
    parser = get_embedding_producer_parameters(parser)

    return parser.parse_args()

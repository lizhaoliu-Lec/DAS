import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from dataset.basic_dataset_scaffold import BaseDataset


def get_dataset(opt, data_path, TrainDatasetClass=None):
    """
    This function generates a training, testing and evaluation dataloader for Metric Learning on the In-Shop Clothes dataset.
    For Metric Learning, training and test sets are provided by one text file, list_eval_partition.txt.
    So no random shuffling of classes.
    """

    # Load train-test-partition text file.
    # ------------- list_eval_partition.txt -------------
    # First Row: number of images
    # Second Row: entry names
    # Rest of the Rows: <image name> <item id> <evaluation status>
    # ---------------------------------------------------
    # In evaluation status:
    # "train" represents training image;
    # "query" represents query image; (test)
    # "gallery" represents gallery image; (evaluation)
    data_info = np.array(
        pd.read_table(data_path + '/Eval/list_eval_partition.txt', header=1, delim_whitespace=True))[1:, :]
    # Separate into training dataset and query/gallery dataset for testing.
    train, query, gallery = data_info[data_info[:, 2] == 'train'][:, :2], \
                            data_info[data_info[:, 2] == 'query'][:, :2], \
                            data_info[data_info[:, 2] == 'gallery'][:, :2]

    # Generate conversions
    lab_conv = {x: i for i, x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in train[:, 1]])))}
    train[:, 1] = np.array([lab_conv[int(x.split('_')[-1])] for x in train[:, 1]])

    lab_conv = {x: i for i, x in enumerate(
        np.unique(np.array([int(x.split('_')[-1]) for x in np.concatenate([query[:, 1], gallery[:, 1]])])))}
    query[:, 1] = np.array([lab_conv[int(x.split('_')[-1])] for x in query[:, 1]])
    gallery[:, 1] = np.array([lab_conv[int(x.split('_')[-1])] for x in gallery[:, 1]])

    # Generate Image-Dicts for training, query and gallery of shape {class_idx:[list of paths to images belong to this class] ...}
    train_image_dict = {}
    for img_path, key in train:
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(data_path + '/' + img_path)

    query_image_dict = {}
    for img_path, key in query:
        if not key in query_image_dict.keys():
            query_image_dict[key] = []
        query_image_dict[key].append(data_path + '/' + img_path)

    gallery_image_dict = {}
    for img_path, key in gallery:
        if not key in gallery_image_dict.keys():
            gallery_image_dict[key] = []
        gallery_image_dict[key].append(data_path + '/' + img_path)

    # train_dataset = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    # eval_dataset = BaseTripletDataset(train_image_dict, opt, is_validation=True)
    # query_dataset = BaseTripletDataset(query_image_dict, opt, is_validation=True)
    # gallery_dataset = BaseTripletDataset(gallery_image_dict, opt, is_validation=True)

    train_dataset = BaseDataset(train_image_dict, opt)
    eval_dataset = BaseDataset(train_image_dict, opt, is_validation=True)
    query_dataset = BaseDataset(query_image_dict, opt, is_validation=True)
    gallery_dataset = BaseDataset(gallery_image_dict, opt, is_validation=True)

    # return {'training': train_dataset,
    #         'testing_query': query_dataset,
    #         'evaluation': eval_dataset,
    #         'testing_gallery': gallery_dataset}

    return {'training': train_dataset,
            'validation': None,
            'testing': query_dataset,
            'evaluation': eval_dataset,
            'evaluation_train': gallery_dataset}

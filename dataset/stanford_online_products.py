import pandas as pd

from dataset.basic_dataset_scaffold import BaseDataset


def get_dataset(opt, data_path, TrainDatasetClass=None):
    image_source_path = data_path + '/images'
    training_files = pd.read_table(opt.source_path + '/Info_Files/Ebay_train.txt', header=0, delimiter=' ')
    test_files = pd.read_table(opt.source_path + '/Info_Files/Ebay_test.txt', header=0, delimiter=' ')

    super_dict = {}
    super_conversion = {}
    for (super_ix, class_ix, image_path) in zip(training_files['super_class_id'],
                                                training_files['class_id'],
                                                training_files['path']):
        if super_ix not in super_dict:
            super_dict[super_ix] = {}
        if class_ix not in super_dict[super_ix]:
            super_dict[super_ix][class_ix] = []
        super_dict[super_ix][class_ix].append(image_source_path + '/' + image_path)

    if opt.use_tv_split:
        if not opt.tv_split_by_samples:
            train_image_dict, val_image_dict = {}, {}
            train_count, val_count = 0, 0
            for super_ix in super_dict.keys():
                class_ixs = sorted(list(super_dict[super_ix].keys()))
                train_val_split = int(len(super_dict[super_ix]) * opt.tv_split_perc)
                train_image_dict[super_ix] = {}
                for _, class_ix in enumerate(class_ixs[:train_val_split]):
                    train_image_dict[super_ix][train_count] = super_dict[super_ix][class_ix]
                    train_count += 1
                val_image_dict[super_ix] = {}
                for _, class_ix in enumerate(class_ixs[train_val_split:]):
                    val_image_dict[super_ix][val_count] = super_dict[super_ix][class_ix]
                    val_count += 1
        else:
            train_image_dict, val_image_dict = {}, {}
            for super_ix in super_dict.keys():
                class_ixs = sorted(list(super_dict[super_ix].keys()))
                train_image_dict[super_ix] = {}
                val_image_dict[super_ix] = {}
                for class_ix in class_ixs:
                    train_val_split = int(len(super_dict[super_ix][class_ix]) * opt.tv_split_perc)
                    train_image_dict[super_ix][class_ix] = super_dict[super_ix][class_ix][:train_val_split]
                    val_image_dict[super_ix][class_ix] = super_dict[super_ix][class_ix][train_val_split:]
    else:
        train_image_dict = super_dict
        val_image_dict = None

    test_image_dict = {}
    train_image_dict_temp = {}
    val_image_dict_temp = {}
    super_train_image_dict = {}
    super_val_image_dict = {}
    train_conversion = {}
    super_train_conversion = {}
    val_conversion = {}
    test_conversion = {}
    super_test_conversion = {}
    class_ix2super_ix = {}

    # Create Training Dictionaries
    for super_ix, super_set in train_image_dict.items():
        super_ix -= 1
        super_train_image_dict[super_ix] = []
        for class_ix, class_set in super_set.items():
            class_ix -= 1
            class_ix2super_ix[class_ix] = super_ix
            super_train_image_dict[super_ix].extend(class_set)
            train_image_dict_temp[class_ix] = class_set
            if class_ix not in train_conversion:
                train_conversion[class_ix] = class_set[0].split('/')[-1].split('_')[0]
                super_conversion[class_ix] = class_set[0].split('/')[-2]
    train_image_dict = train_image_dict_temp

    # Create Validation Dictionaries
    if opt.use_tv_split:
        for super_ix, super_set in val_image_dict.items():
            super_ix -= 1
            super_val_image_dict[super_ix] = []
            for class_ix, class_set in super_set.items():
                class_ix -= 1
                super_val_image_dict[super_ix].extend(class_set)
                val_image_dict_temp[class_ix] = class_set
                if class_ix not in val_conversion:
                    val_conversion[class_ix] = class_set[0].split('/')[-1].split('_')[0]
                    super_conversion[class_ix] = class_set[0].split('/')[-2]
        val_image_dict = val_image_dict_temp
    else:
        val_image_dict = None

    # Create Test Dictionaries
    for class_ix, img_path in zip(test_files['class_id'], test_files['path']):
        class_ix = class_ix - 1
        if class_ix not in test_image_dict.keys():
            test_image_dict[class_ix] = []
        test_image_dict[class_ix].append(image_source_path + '/' + img_path)
        test_conversion[class_ix] = img_path.split('/')[-1].split('_')[0]
        super_test_conversion[class_ix] = img_path.split('/')[-2]

    if val_image_dict:
        val_dataset = BaseDataset(val_image_dict, opt, is_validation=True)
        val_dataset.conversion = val_conversion
    else:
        val_dataset = None

    print('\nDataset Setup:\nUsing Train-Val Split: {0}\n#Classes: Train ({1}) | Val ({2}) | Test ({3})\n'.format(
        opt.use_tv_split, len(train_image_dict), len(val_image_dict) if val_image_dict else 'X', len(test_image_dict)))

    super_train_dataset = BaseDataset(super_train_image_dict, opt, is_validation=True)
    if TrainDatasetClass is None:
        TrainDatasetClass = BaseDataset
    train_dataset = TrainDatasetClass(train_image_dict, opt, class_ix2super_ix=class_ix2super_ix)
    test_dataset = BaseDataset(test_image_dict, opt, is_validation=True)
    eval_dataset = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset = BaseDataset(train_image_dict, opt)

    super_train_dataset.conversion = super_train_conversion
    train_dataset.conversion = train_conversion
    test_dataset.conversion = test_conversion
    eval_dataset.conversion = train_conversion

    # print("===> train_dataset.conversion: {}".format(train_dataset.conversion))
    # print("===> test_dataset.conversion: {}".format(test_dataset.conversion))
    # print("===> train_dataset.conversion.keys.unique: {}".format(set(train_dataset.conversion.keys())))
    # print("===> train_dataset.conversion.values.unique: {}".format(set(train_dataset.conversion.values())))
    # print("===> test_dataset.conversion.keys.unique: {}".format(set(test_dataset.conversion.keys())))
    # print("===> test_dataset.conversion.values.unique: {}".format(set(test_dataset.conversion.values())))
    # print("===> class_ix2super_ix: {}".format(class_ix2super_ix))

    return {'training': train_dataset,
            'validation': val_dataset,
            'testing': test_dataset,
            'evaluation': eval_dataset,
            'evaluation_train': eval_train_dataset,
            'super_evaluation': super_train_dataset}

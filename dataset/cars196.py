import os
import pickle
import numpy as np

from dataset.basic_dataset_scaffold import BaseDataset

def get_default_image_dict_and_conversion(opt, data_path):
    image_source_path = data_path + '/images'
    image_classes = sorted([x for x in os.listdir(image_source_path)])
    total_conversion = {i: x for i, x in enumerate(image_classes)}
    image_list = {
        i: sorted([image_source_path + '/' + key + '/' + x for x in os.listdir(image_source_path + '/' + key)])
        for i, key in enumerate(image_classes)}
    image_list = [[(key, img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list = [x for y in image_list for x in y]

    # Dictionary of structure class:list_of_samples_with_said_class
    image_dict = {}
    for key, img_path in image_list:
        if key not in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    # Use the first half of the sorted data as training and the second half as test set
    keys = sorted(list(image_dict.keys()))
    train, test = keys[:len(keys) // 2], keys[len(keys) // 2:]

    # If required, split the training data into a train/val setup either by or per class.
    if opt.use_tv_split:
        if not opt.tv_split_by_samples:
            train_val_split = int(len(train) * opt.tv_split_perc)
            train, val = train[:train_val_split], train[train_val_split:]
            train_image_dict = {i: image_dict[key] for i, key in enumerate(train)}
            val_image_dict = {i: image_dict[key] for i, key in enumerate(val)}
        else:
            val = train
            train_image_dict, val_image_dict = {}, {}
            for key in train:
                train_ixs = np.random.choice(len(image_dict[key]), int(len(image_dict[key]) * opt.tv_split_perc),
                                             replace=False)
                val_ixs = np.array([x for x in range(len(image_dict[key])) if x not in train_ixs])
                train_image_dict[key] = np.array(image_dict[key])[train_ixs]
                val_image_dict[key] = np.array(image_dict[key])[val_ixs]
        val_dataset = BaseDataset(val_image_dict, opt, is_validation=True)
        val_conversion = {i: total_conversion[key] for i, key in enumerate(val)}
        val_dataset.conversion = val_conversion
    else:
        train_image_dict = {key: image_dict[key] for key in train}
        val_image_dict = None
        val_dataset = None

    train_conversion = {i: total_conversion[key] for i, key in enumerate(train)}
    test_conversion = {i: total_conversion[key] for i, key in enumerate(test)}

    test_image_dict = {key: image_dict[key] for key in test}
    
    return (
        val_dataset,
        train_image_dict, val_image_dict, test_image_dict,
        train_conversion, test_conversion
    )


def get_ooDML_image_dict_and_conversion(opt, data_path):
    
    val_dataset = None
    val_image_dict = None
    
    with open(opt.data_split_path, "rb") as f:
        oodml_split = pickle.load(f)
    
    data_split_id = opt.data_split_id
    
    meta_data = oodml_split[data_split_id]
    
    train_classnames = meta_data["train"]
    test_classnames = meta_data["test"]
    
    train_image_dict = {}
    train_conversion = {}
    test_image_dict = {}
    test_conversion = {}
    
    class_idx = 0
    train_img_idx = 0
    for train_class_idx, train_classname in enumerate(train_classnames):
        train_conversion[train_class_idx] = train_classname
        
        if class_idx not in train_image_dict:
            train_image_dict[class_idx] = []
        class_img_path = os.path.join(data_path, "images", train_classname)
        for img_name in os.listdir(class_img_path):
            train_image_dict[class_idx].append(
                os.path.join(class_img_path, img_name)
            )                
            train_img_idx += 1
        sorted(train_image_dict[class_idx])
        class_idx += 1
        
    test_img_idx = 0
    for test_class_idx, test_classname in enumerate(test_classnames):
        test_conversion[test_class_idx] = test_classname
        
        if class_idx not in test_image_dict:
            test_image_dict[class_idx] = []
        class_img_path = os.path.join(data_path, "images", test_classname)
        for img_name in os.listdir(class_img_path):
            test_image_dict[class_idx].append(
                os.path.join(class_img_path, img_name)
            )
            test_img_idx += 1
        sorted(test_image_dict[class_idx])
        class_idx += 1
            
    return (
        val_dataset,
        train_image_dict, val_image_dict, test_image_dict,
        train_conversion, test_conversion
    )

def get_dataset(opt, data_path, TrainDatasetClass=None):
    if opt.data_split == "default":
        print("Using default split...")
        (
            val_dataset,
            train_image_dict, val_image_dict, test_image_dict,
            train_conversion, test_conversion
        ) = get_default_image_dict_and_conversion(opt, data_path)
    elif opt.data_split == "ooDML":
        print("Using ooDML split with split_id={}".format(opt.data_split_id))
        (
            val_dataset,
            train_image_dict, val_image_dict, test_image_dict,
            train_conversion, test_conversion
        ) = get_ooDML_image_dict_and_conversion(opt, data_path)
    else:
        raise ValueError("Unsupported data split {}".format(opt.data_split))
    
    print('\nDataset Setup:\nUsing Train-Val Split: {0}\n#Classes: Train ({1}) | Val ({2}) | Test ({3})\n'.format(
        opt.use_tv_split, len(train_image_dict), len(val_image_dict) if val_image_dict else 'X', len(test_image_dict)))

    if TrainDatasetClass is None:
        TrainDatasetClass = BaseDataset
    train_dataset = TrainDatasetClass(train_image_dict, opt)
    test_dataset = BaseDataset(test_image_dict, opt, is_validation=True)
    eval_dataset = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset = BaseDataset(train_image_dict, opt, is_validation=False)
    train_dataset.conversion = train_conversion
    test_dataset.conversion = test_conversion
    eval_dataset.conversion = test_conversion
    eval_train_dataset.conversion = train_conversion

    return {'training': train_dataset,
            'validation': val_dataset,
            'testing': test_dataset,
            'evaluation': eval_dataset,
            'evaluation_train': eval_train_dataset}

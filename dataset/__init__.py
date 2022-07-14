import dataset.cars196
import dataset.cub200
import dataset.stanford_online_products
import dataset.in_shop


def select(dataset, opt, data_path, TrainDatasetClass=None):
    if 'cub200' in dataset:
        return cub200.get_dataset(opt, data_path, TrainDatasetClass)

    if 'cars196' in dataset:
        return cars196.get_dataset(opt, data_path, TrainDatasetClass)

    if 'online_products' in dataset:
        return stanford_online_products.get_dataset(opt, data_path, TrainDatasetClass)

    if 'in-shop' in dataset:
        return in_shop.get_dataset(opt, data_path, TrainDatasetClass)

    raise NotImplementedError('A dataset for {} is currently not implemented.\n\
                               Currently available are : cub200, cars196 & online_products!'.format(dataset))

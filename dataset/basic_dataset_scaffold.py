import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    BASIC PYTORCH DATASET USED FOR ALL DATASETS
    """

    def __init__(self, image_dict, opt, is_validation=False, class_ix2super_ix=None):
        self.is_validation = is_validation
        self.pars = opt

        self.image_dict = image_dict

        self.n_files = None
        self.avail_classes = None
        self.image_list = None
        self.image_paths = None
        self.is_init = False
        self.class_ix2super_ix = class_ix2super_ix

        self.init_setup()

        if 'bninception' not in opt.arch:
            self.f_norm = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            # normalize = transforms.Normalize(mean=[0.502, 0.4588, 0.4078],std=[1., 1., 1.])
            self.f_norm = normalize = transforms.Normalize(mean=[0.502, 0.4588, 0.4078], std=[0.0039, 0.0039, 0.0039])

        self.crop_size = crop_im_size = 224 if 'googlenet' not in opt.arch else 227
        if opt.augmentation == 'big':
            crop_im_size = 256

        self.normal_transform = []
        if not self.is_validation:
            if opt.augmentation == 'base' or opt.augmentation == 'big':
                self.normal_transform.extend(
                    [transforms.RandomResizedCrop(size=crop_im_size)])
            elif opt.augmentation == 'adv':
                self.normal_transform.extend(
                    [transforms.RandomResizedCrop(size=crop_im_size),
                     transforms.RandomGrayscale(p=0.2),
                     transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)])
            elif opt.augmentation == 'red':
                self.normal_transform.extend(
                    [transforms.Resize(size=256),
                     transforms.RandomCrop(crop_im_size)])
            self.normal_transform.append(transforms.RandomHorizontalFlip(0.5))

            # if set, no image augmentation will be used, except the fixed resize
            if opt.no_random_augmentation:
                self.normal_transform = [transforms.Resize(size=(crop_im_size, crop_im_size))]
        else:
            self.normal_transform.extend([transforms.Resize(256), transforms.CenterCrop(crop_im_size)])
        self.normal_transform.extend([transforms.ToTensor(), normalize])
        self.normal_transform = transforms.Compose(self.normal_transform)

    def init_setup(self):
        self.n_files = np.sum([len(self.image_dict[label]) for label in self.image_dict.keys()])
        self.avail_classes = sorted(list(self.image_dict.keys()))

        counter = 0
        temp_image_dict = {}
        for i, label in enumerate(self.avail_classes):
            temp_image_dict[label] = []
            for path in self.image_dict[label]:
                temp_image_dict[label].append([path, counter])
                counter += 1

        self.image_dict = temp_image_dict
        self.image_list = [[(img_counter[0], label) for img_counter in self.image_dict[label]] for label in
                           self.image_dict.keys()]
        self.image_list = [img for class_images in self.image_list for img in class_images]

        self.image_paths = self.image_list

        self.is_init = True

    @staticmethod
    def ensure_3dim(img):
        if len(img.size) == 2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        input_image = self.ensure_3dim(Image.open(self.image_list[idx][0]))

        # Basic preprocessing.
        im_a = self.normal_transform(input_image)
        if 'bninception' in self.pars.arch:
            im_a = im_a[range(3)[::-1], :]
        if self.class_ix2super_ix is None:
            return self.image_list[idx][-1], im_a, idx, -1  # FIXME -1 for no super class
        else:
            return self.image_list[idx][-1], im_a, idx, self.class_ix2super_ix[self.image_list[idx][-1]]

    def __len__(self):
        return self.n_files

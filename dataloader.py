# from .vision import VisionDataset
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

from PIL import Image
import numpy as np
import os.path
import sys
import torch
import pickle

import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as Transforms

random_seed = 888

def vgg_data_loader(root, batch_size, resize_size, crop_size):
    transform_list = []

    transform_list += [Transforms.Resize((resize_size, resize_size))]
    transform_list += [Transforms.RandomCrop((crop_size, crop_size))]
    # PIL -> Tensor
    transform_list += [Transforms.ToTensor(),
                       Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                            std=(0.5, 0.5, 0.5))]

    transform = Transforms.Compose(transform_list)

    dataset = ImageFolder(root, transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    cls_num = len(os.listdir(root))

    return dataloader, len(dataset), cls_num


def omniglot_data_loader(root, batch_size, resize_size, crop_size):
    omniglot_transformer = Transforms.Compose([Transforms.Resize((resize_size, resize_size)),
                                               Transforms.RandomCrop((crop_size, crop_size)),
                                               Transforms.Grayscale(num_output_channels=1),
                                               Transforms.ToTensor(),
                                               Transforms.Normalize(mean=[0.5],
                                                                    std=[-0.5])])

    dataset = ImageFolder(root, transform=omniglot_transformer)
    cls_num = len(os.listdir(root))

    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader, len(dataset), cls_num


def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


class DatasetFolder(VisionDataset):
    def __init__(self, img_path_pkl, dtype, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):

        # 여기서 (img_path, class)의 튜플로 만들어진 리스트 samples 생성
        with open(img_path_pkl, 'rb') as f:
            img_path_dict = pickle.load(f)
            samples = img_path_dict[dtype]

        # root가 필요하긴 해서 강제로 가져옴
        # root = '/'.join(samples[0][0].split('/')[:7])
        root = '/'.join(samples[0][0].split('/')[:4])
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                extensions)))

        self.loader = loader
        self.extensions = extensions

        # self.classes = classes
        # self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImagenetDataset(DatasetFolder):
    '''
    < dtype >

    SMALL-BACKGROUND
    SMALL-BACKGROUND-COMP
    SMALL-EVAL
    LARGE
    '''

    def __init__(self, img_path_pkl, dtype, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(ImagenetDataset, self).__init__(img_path_pkl,
                                              dtype,
                                              loader,
                                              IMG_EXTENSIONS if is_valid_file is None else None,
                                              transform=transform,
                                              target_transform=target_transform,
                                              is_valid_file=is_valid_file)
        self.imgs = self.samples


class ImagenetTransform(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if F._is_pil_image(img):
            w, h = img.size
        elif isinstance(img, torch.Tensor) and img.dim() > 2:
            w, h = img.shape[-2:][::-1]
        else:
            raise TypeError("Unexpected type {}".format(type(img)))

        short_side = h if h < w else w
        crop_size = short_side

        # Crop the center
        i = (h - crop_size) // 2
        j = (w - crop_size) // 2

        # -1 ~ 1 후에 random noise 추가
        img = F.resize(F.crop(img, i, j, crop_size, crop_size), (self.size, self.size))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


def _init_fn():
    np.random.seed(random_seed)


def data_loader(img_path_pkl, dtype='SMALL-BACKGROUND',
                batch_size=16, resize_size=84, crop_size=64):
    transform_list = []
    transform_list += [ImagenetTransform(size=resize_size),
                       Transforms.RandomCrop(size=crop_size),
                       Transforms.ToTensor(),
                       Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                            std=(0.5, 0.5, 0.5))]

    transform = Transforms.Compose(transform_list)

    dataset = ImagenetDataset(img_path_pkl, dtype, transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0,
                            worker_init_fn=_init_fn)

    if dtype == 'SMALL-BACKGROUND' or dtype == 'SMALL-BACKGROUND-COMP':
        cls_num = int(img_path_pkl.split('_')[2])
    elif dtype == 'SMALL-EVAL':
        cls_num = int(img_path_pkl.split('_')[4])
    elif dtype == 'LARGE':
        cls_num = int(img_path_pkl.split('_')[6])

    return dataloader, len(dataset), cls_num
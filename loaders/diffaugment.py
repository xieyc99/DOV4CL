import functools
from torch.utils.data import DataLoader, TensorDataset
import os
import random
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image, ImageFilter
import pandas as pd
from functools import partial
from torch import  Tensor
import glob
from typing import Callable, Tuple
import torch
import torch.nn as nn

from torch.utils.data import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import  VisionDataset
from kornia import augmentation as aug
from methods.MOCOv3.mocov3 import GaussianBlur

import natsort
from utils.util import *
import methods.DINO.utils as utils

class Subset(torch.utils.data.Subset):
    """Overwrite subset class to provide class methods of main class."""

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # imagenet
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # cifar10
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            # transforms.RandomResizedCrop(32, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            # transforms.RandomResizedCrop(32, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            # transforms.RandomResizedCrop(16, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

class PoisonAgent():
    def __init__(self, args, fre_agent, trainset, validset, memory_loader, magnitude):
        self.args = args
        self.trainset = trainset
        self.validset = validset
        self.memory_loader = memory_loader
        self.poison_num = int(len(trainset) * self.args.poison_ratio)
        self.fre_poison_agent = fre_agent

        self.magnitude = magnitude

        self.construct_experiment()


    def construct_experiment(self):
        if self.args.poisonkey is None:
            init_seed = np.random.randint(0, 2 ** 32 - 1)
        else:
            init_seed = int(self.args.poisonkey)

        np.random.seed(init_seed)
        print(f'Initializing Poison data (chosen images, examples, sources, labels) with random seed {init_seed}')
        self.train_pos_loader, self.test_loader, self.test_pos_loader, self.memory_loader  = self.choose_poisons_randomly()




    def choose_poisons_randomly(self):

        #construct class prototype for each class


        x_train_np, x_test_np = self.trainset.data.astype(np.float32) / 255., self.validset.data.astype(
            np.float32) / 255.

        x_memory_np =  self.memory_loader.dataset.data.astype(np.float32) / 255.



        y_train_np, y_test_np = np.array(self.trainset.targets), np.array(self.validset.targets)
        y_memory_np = np.array(self.memory_loader.dataset.targets)

        x_train_tensor, y_train_tensor = torch.tensor(x_train_np), torch.tensor(y_train_np, dtype=torch.long)
        x_test_tensor, y_test_tensor = torch.tensor(x_test_np), torch.tensor(y_test_np, dtype=torch.long)

        y_memory_tensor = torch.tensor(y_memory_np, dtype=torch.long)
        x_memory_tensor = torch.tensor(x_memory_np)


        x_train_tensor = x_train_tensor.permute(0, 3, 1, 2)
        x_test_tensor = x_test_tensor.permute(0, 3, 1, 2)
        x_memory_tensor = x_memory_tensor.permute(0, 3, 1, 2)

        x_train_origin = x_train_tensor.clone().detach()




        poison_index = torch.where(y_train_tensor == self.args.target_class)[0]
        poison_index = poison_index[:self.poison_num]


        if self.args.threat_model == 'our':
            x_train_tensor[poison_index], y_train_tensor[poison_index] = self.fre_poison_agent.Poison_Frequency_Diff(x_train_tensor[poison_index], y_train_tensor[poison_index], self.magnitude)
            x_test_pos_tensor, y_test_pos_tensor = self.fre_poison_agent.Poison_Frequency_Diff(x_test_tensor.clone().detach(), y_test_tensor.clone().detach(), self.magnitude)


        else:
            raise  NotImplementedError




        # index = poison_index[0]
        #
        # show_example = torch.cat([x_train_origin[index:index + 1], x_train_tensor[index:index + 1]], dim=0)
        # view1 = individual_transform(show_example)
        # view2 = individual_transform(show_example)

        y_test_pos_tensor = torch.ones_like(y_test_pos_tensor, dtype=torch.long) * self.args.target_class

        train_index =   torch.tensor(list(range(len(self.trainset))), dtype = torch.long)
        test_index =    torch.tensor(list(range(len(self.validset))), dtype = torch.long)
        memory_index = torch.tensor(list(range(len(x_memory_tensor))), dtype = torch.long)


        train_sampler = None


        train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor, train_index), batch_size=self.args.batch_size, sampler=train_sampler, shuffle= (train_sampler is None), drop_last=True)
        test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor, test_index), batch_size=self.args.eval_batch_size, shuffle=False, drop_last=True)
        test_pos_loader = DataLoader(TensorDataset(x_test_pos_tensor, y_test_pos_tensor, test_index), batch_size=self.args.eval_batch_size, shuffle=False)
        memory_loader = DataLoader(TensorDataset(x_memory_tensor, y_memory_tensor, memory_index), batch_size=self.args.eval_batch_size, shuffle=False)



        return train_loader, test_loader, test_pos_loader, memory_loader





class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x if random.random() > self.p else self.fn(x)



def set_aug_diff(args):
    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        args.size = 32
        args.num_classes = 10
        args.save_freq = 100

    elif args.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        args.size = 32
        args.num_classes = 100
        args.save_freq = 100

    elif args.dataset == 'svhn':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        args.size = 32
        args.num_classes = 10
        args.save_freq = 100

    elif 'image' in args.dataset and 'tiny' not in args.dataset:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        args.size = 224
        args.save_freq = 100
        args.num_classes = 10
    
    elif args.dataset == 'tiny-imagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
        args.size = 64
        args.save_freq = 100
        args.num_classes = 200

    elif args.dataset == 'stl10':
        mean = (0.507, 0.487, 0.441)
        std = (0.267, 0.256, 0.276)
        args.size = 96
        args.save_freq = 100
        args.num_classes = 10

    else:
        raise ValueError(args.dataset)

    normalize = aug.Normalize(mean=mean, std=std)

    ####################### Define Diff Transforms #######################

    if 'cifar' in args.dataset  or 'svhn' in args.dataset or 'image' in args.dataset or 'stl10' in args.dataset:

        if not args.disable_normalize:
            if args.method == 'mocov3':
                train_transform = nn.Sequential(
                        aug.RandomResizedCrop(size=(args.size, args.size), scale=(0.2, 1.)),
                        RandomApply(aug.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
                        aug.RandomGrayscale(p=0.2),
                        RandomApply(GaussianBlur(), p=0.5),
                        aug.RandomHorizontalFlip(),
                        normalize
                )
                ft_transform = nn.Sequential( aug.RandomResizedCrop(size=(args.size, args.size), scale=(0.2, 1.)),
                                                    aug.RandomHorizontalFlip(),
                                                    aug.RandomGrayscale(p=0.2),
                                                    normalize)

                test_transform =  nn.Sequential(normalize)
            else:
                train_transform = nn.Sequential( aug.RandomResizedCrop(size = (args.size, args.size), scale=(0.2, 1.0)),
                                                    aug.RandomHorizontalFlip(),
                                                    RandomApply(aug.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
                                                    aug.RandomGrayscale(p=0.2),
                                                    normalize)

                ft_transform = nn.Sequential( aug.RandomResizedCrop(size=(args.size, args.size), scale=(0.2, 1.)),
                                                    aug.RandomHorizontalFlip(),
                                                    aug.RandomGrayscale(p=0.2),
                                                    normalize)

                test_transform =  nn.Sequential(normalize)

        else:
            train_transform = nn.Sequential(aug.RandomResizedCrop(size=(args.size, args.size), scale=(0.2, 1.0)),
                                            aug.RandomHorizontalFlip(),
                                            RandomApply(aug.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
                                            aug.RandomGrayscale(p=0.2),
                                            )


            ft_transform = nn.Sequential(aug.RandomResizedCrop(size=(args.size, args.size), scale=(0.2, 1.)),
                                            aug.RandomHorizontalFlip(),
                                            aug.RandomGrayscale(p=0.2),
                                            )

            test_transform = nn.Sequential(
                                            nn.Identity(),
                                        )

    ####################### Define Load Transform ####################
    if args.method == 'mae':
        transform_load = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    elif args.method == 'dino':
        transform_load = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    elif 'cifar' in args.dataset or 'svhn' in args.dataset or 'image' in args.dataset or 'stl10' in args.dataset:
        transform_load = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean,std) if not args.disable_normalize else transforms.Lambda(lambda x: x)
        ])
    else:
         raise  NotImplementedError


    ####################### Define Datasets #######################
    if args.dataset == 'cifar10':

        train_dataset = CIFAR10(root=args.data_path,
                                         train=True,
                                         transform=transform_load,
                                         download=True)
        
        ft_dataset = CIFAR10(root=args.data_path,
                                      transform=transform_load,
                                      download=False)
        test_dataset = CIFAR10(root=args.data_path,
                                        train=False,
                                        transform=transform_load,
                                        download=True)
        memory_dataset = CIFAR10(root=args.data_path,
                                          train=True,
                                          transform=transform_load,
                                          download=False)


    elif args.dataset == 'cifar100':
        train_dataset = CIFAR100(root=args.data_path,
                                          train = True,
                                          transform=transform_load,
                                          download=True)
        ft_dataset = CIFAR100(root=args.data_path,
                                       transform=transform_load,
                                       download=False)
        test_dataset = CIFAR100(root=args.data_path,
                                         train=False,
                                         transform=transform_load,
                                         download=True)
        memory_dataset = CIFAR100(root=args.data_path,
                                           train=True,
                                           transform=transform_load,
                                           download=False)
    elif args.dataset == 'svhn':
        train_dataset = SVHN(root=args.data_path+'/SVHN', split='train', transform=transform_load, download=True)
        ft_dataset = SVHN(root=args.data_path+'/SVHN', split='test', transform=transform_load, download=False)
        test_dataset = SVHN(root=args.data_path+'/SVHN', split='test', transform=transform_load, download=True)
        memory_dataset = SVHN(root=args.data_path+'/SVHN', split='train', transform=transform_load, download=False)

    elif 'image' in args.dataset and 'tiny' not in args.dataset:
        train_dataset = ImageNet(root=args.data_path+'/'+args.dataset, train = True, transform=transform_load)
        ft_dataset = ImageNet(root=args.data_path+'/'+args.dataset, transform=transform_load)
        test_dataset = ImageNet(root=args.data_path+'/'+args.dataset, train=False, transform=transform_load)
        memory_dataset = ImageNet(root=args.data_path+'/'+args.dataset, train=True, transform=transform_load)

    elif args.dataset == 'tiny-imagenet':
        train_dataset = TinyImageNet(root=args.data_path+'/tiny-imagenet-200/', train = True)
        ft_dataset = TinyImageNet(root=args.data_path+'/tiny-imagenet-200/')
        test_dataset = TinyImageNet(root=args.data_path+'/tiny-imagenet-200/', train=False,)
        memory_dataset = TinyImageNet(root=args.data_path+'/tiny-imagenet-200/', train=True)
    elif args.dataset == 'stl10':
        train_dataset = STL10(root=args.data_path, split='train+unlabeled', transform=transform_load, download=False)
        ft_dataset = STL10(root=args.data_path, split='test', transform=transform_load, download=False)
        test_dataset = STL10(root=args.data_path, split='test', transform=transform_load, download=False)
        memory_dataset = STL10(root=args.data_path, split='train', transform=transform_load, download=False)
    else:

         raise NotImplementedError
    
    if args.part != '':
        subset1, subset2 = split_dataset(train_dataset, args.num_classes, args.unlabel, args.ratio)
        if args.part == 'before':
            train_dataset = subset1
        elif args.part == 'after':
            train_dataset = subset2

    train_sampler = None
    ft_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    ft_loader = torch.utils.data.DataLoader(
        ft_dataset, batch_size=args.eval_batch_size, shuffle=(ft_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=ft_sampler)

    # indices  = np.random.choice(len(test_dataset), 1024, replace=False)
    # sampler = SubsetRandomSampler(indices)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=None)

    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    return train_loader, train_sampler, train_dataset, ft_loader, ft_sampler, test_loader, test_dataset, memory_loader, train_transform, ft_transform, test_transform



class CIFAR10(datasets.CIFAR10):
    """Super-class CIFAR10 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img,  target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index



class CIFAR100(datasets.CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index

class SVHN(datasets.SVHN):

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.transpose(1,2,0))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # target = torch.tensor(target)

        return img, target, index

    def get_target(self, index):
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index
    
class STL10(datasets.STL10):

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.transpose(1,2,0))

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        target = torch.tensor(target)
        # print(type(img))
        # print(type(target))

        return img, target, index

    def get_target(self, index):
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index

class ImageNet(datasets.ImageFolder):
    def __init__(self, root, train=True, transform=None):
        if train:
            self.root_dir = root+'/train'
        else:
            self.root_dir = root+'/val'
        if transform is None:
            self.transform = transforms.Compose([
                            transforms.Resize(256),  
                            transforms.CenterCrop(224),  
                            transforms.ToTensor(),  
                            # transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                            # std=[0.229, 0.224, 0.225])
                        ])
        else:
            self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)

        self.imgs, self.targets = self._make_dataset(self.root_dir)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, target = self.imgs[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target, index

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, dir):
        images = []
        targets = []
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.endswith(".JPEG") or fname.endswith(".png"):
                        path = os.path.join(root, fname)
                        images.append(path)
                        targets.append(self.class_to_idx[target])
        return images, targets

class TinyImageNet(datasets.ImageFolder):
    def __init__(self, root, train=True):
        if train:
            self.root_dir = root+'/train'
        else:
            self.root_dir = root+'/val'
        self.transform = transforms.Compose([
                        transforms.ToTensor(),  # 将图像转换为张量，并将像素值缩放到 [0, 1]
                    ])
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)

        self.imgs, self.targets = self._make_dataset(self.root_dir)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, target = self.imgs[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target, index

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, dir):
        images = []
        targets = []
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.endswith(".JPEG") or fname.endswith(".png"):
                        path = os.path.join(root, fname)
                        images.append(path)
                        targets.append(self.class_to_idx[target])
        return images, targets

def split_dataset(train_dataset, num_classes, unlabel=True, ratio=0.5):

    split_dataset1 = []
    split_dataset2 = []

    if not unlabel:
        print('split using label')
        class_indices = [[] for _ in range(num_classes)]

        for i, (image, label, index) in enumerate(train_dataset):
            class_indices[label].append(i)

        for i, indices in enumerate(class_indices):
            np.random.shuffle(indices)
            if i == 0:
                print(indices[:10])
            
            split_index = int(len(indices)*ratio)
            
            split_dataset1.extend(indices[:split_index])
            split_dataset2.extend(indices[split_index:])
    else:
        print('split random')
        indices = np.random.choice(len(train_dataset), int(len(train_dataset)*ratio), replace=False)
        indices = indices.tolist()
        print(indices[:10])
        for i in range(len(train_dataset)):
            if i in indices:
                split_dataset1.append(i)
            else:
                split_dataset2.append(i)
        print('split_dataset1:', len(split_dataset1))
        print('split_dataset2:', len(split_dataset2))

    subset1 = Subset(train_dataset, split_dataset1)
    subset2 = Subset(train_dataset, split_dataset2)

    return subset1, subset2

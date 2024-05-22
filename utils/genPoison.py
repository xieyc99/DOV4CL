from utils.frequency import PoisonFre
from loaders.diffaugment import set_aug_diff, PoisonAgent
import argparse
import os
from PIL import Image
import numpy as np 

import torchvision.transforms as transforms
import torchvision.datasets as datasets

parser = argparse.ArgumentParser(description='CTRL genPoison')

parser.add_argument('--data_path', default='D:\Exp\datasets\imagenet100')  # imagenet100
parser.add_argument('--dataset', default='imagenet100', choices=['cifar10', 'cifar100', 'imagenet100'])
parser.add_argument('--num_classes', default=100)
parser.add_argument('--image_size', default = 224, type=int)
parser.add_argument('--disable_normalize', action='store_true', default=True)
parser.add_argument('--full_dataset', action='store_true', default=True)
parser.add_argument('--window_size', default = 224, type=int)
parser.add_argument('--eval_batch_size', default = 128, type=int)
parser.add_argument('--num_workers', default=4, type=int)

### training
parser.add_argument('--arch', default='resnet18', type=str, choices=['resnet18', 'resnet50', 'resnet101', 'shufflenet', 'mobilenet', 'squeezenet'])
parser.add_argument('--batch_size', default = 128, type=int)
parser.add_argument('--epochs', default = 600, type=int)
parser.add_argument('--eval_epoch', default=50, type=int, metavar='N',
                    help='how often to evaluate and save when train')
parser.add_argument('--start_epoch', default = 0, type=int)
parser.add_argument('--remove', default = 'none', choices=['crop', 'flip', 'color', 'gray', 'none'])
parser.add_argument('--update_model', action='store_true', default=False)
parser.add_argument('--contrastive', action='store_true', default=False)
parser.add_argument('--distill_freq', default=5, type=int)
parser.add_argument('--saved_path', default=None, type=str, metavar='PATH')
parser.add_argument('--mode', default='normal', choices=['normal', 'frequency'])
parser.add_argument('--trial', default='clean', type=str, choices=['clean', 'poisoned'])
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--schedule', type=int, nargs='+', default=[600, 720],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--lr', default=0.06, type=float)
parser.add_argument('--wd', default=5e-4, type=float)
parser.add_argument('--temp', default=0.5, type=float)

###poisoning
parser.add_argument('--poisonkey', default=7777, type=int)
parser.add_argument('--target_class', default=0, type=int)
parser.add_argument('--poison_ratio', default = 0.01, type=float)
parser.add_argument('--pin_memory', action='store_true', default=False)
parser.add_argument('--select', action='store_true', default=False)
parser.add_argument('--reverse', action='store_true', default=False)
parser.add_argument('--trigger_position', default=[15,31], nargs ='+', type=int)
parser.add_argument('--magnitude', default = 100.0, type=float)
parser.add_argument('--trigger_size', default=5, type=int)
parser.add_argument('--channel', default=[1,2], nargs ='+', type=int)
parser.add_argument('--threat_model', default='our', choices=['our'])
parser.add_argument('--loss_alpha', default = 2.0, type=float)
parser.add_argument('--strength', default= 1.0, type=float)  ### augmentation strength

###logging
parser.add_argument('--log_path', default='Experiments_SL', type=str, help='path to save log')
parser.add_argument('--poison_knn_eval_freq', default=5, type=int)
parser.add_argument('--poison_knn_eval_freq_iter', default=1, type=int)
parser.add_argument('--debug', action='store_true', default=False)

###others
parser.add_argument('--distributed', action='store_true',
                    help='distributed training')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

args = parser.parse_args()

args.trigger_position = [int(0.5*(args.window_size-1)),int(args.window_size-1)]

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
args.size = 64
args.save_freq = 100
args.num_classes = 100

transform_load = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std) if not args.disable_normalize else transforms.Lambda(lambda x: x)])

save_path = f'D:\Exp\datasets\{args.dataset}-ctrl_poisoned-{args.target_class}-{args.poison_ratio}-{args.trigger_position}-{args.magnitude}'
if not os.path.exists(save_path):
    os.mkdir(save_path)

# create data loader
    train_loader, train_sampler, train_dataset, ft_loader, ft_sampler, test_loader, test_dataset, memory_loader, train_transform, ft_transform, test_transform = set_aug_diff(args)

class ImageNet(datasets.ImageFolder):
    def __init__(self, root_dir, train, transform=None):
        # super().__init__(root_dir, transform=None)
        if train:
            self.root_dir = root_dir+'/train'
        else:
            self.root_dir = root_dir+'/val'
        self.transform = transform
        self.classes, self.class_to_idx, self.idx_to_class = self._find_classes(self.root_dir)

        self.imgs = self._make_dataset(self.root_dir)
        # self.data, self.targets = self._make_rawdata(self.root_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, target = self.imgs[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target, index

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
        return classes, class_to_idx, idx_to_class

    def _make_dataset(self, dir):
        images = []
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.endswith(".JPEG") or fname.endswith(".png"):
                        path = os.path.join(root, fname)
                        item = (path, self.class_to_idx[target])
                        images.append(item)
        return images

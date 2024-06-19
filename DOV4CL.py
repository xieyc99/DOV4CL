from torch.utils.data import TensorDataset, RandomSampler
import torch
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
from utils.util import *
from loaders.diffaugment import *

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
import random

import torch
from sklearn.decomposition import PCA

from networks.resnet_cifar import model_dict as model_dict_cifar
from networks.resnet_org import model_dict
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from scipy import stats
from utils.pos_embed import interpolate_pos_embed
from networks.vit_cifar import *
from methods.DINO.utils import Solarize
from mmselfsup.models.backbones import ResNet, MoCoV3ViT
import argparse

parser = argparse.ArgumentParser(description='CTRL Training')

parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--seed', default=99, type=int, help='seed for initializing training. ')
parser.add_argument('--n_sample_train', default=32, type=int, help='N_public')
parser.add_argument('--n_sample_test', default=32, type=int, help='N_private')
parser.add_argument('--n_aug', default=2, type=int, help='m')
parser.add_argument('--n_aug_local', default=6, type=int, help='n')
parser.add_argument('--n_epoch', default=50, type=int, help='T')
parser.add_argument('--lamda', default=1, type=int, help='α,β')
parser.add_argument('--D_public', default='cifar10', type=str, help='public dataset of defender')
parser.add_argument('--M_shadow_arch', default='resnet18', type=str, help='the encoder architecture of M_shadow')
parser.add_argument('--M_shadow_dataset', default='resnet18', type=str, help='the training set of M_shadow')
parser.add_argument('--M_shadow_path', default='', type=str, help='the path of M_shadow')
parser.add_argument('--M_suspect_arch', default='resnet18', type=str, help='the encoder architecture of M_suspect')
parser.add_argument('--M_suspect_dataset', default='resnet18', type=str, help='the training set of M_suspect')
parser.add_argument('--M_suspect_path', default='', type=str, help='the path of M_suspect')

args = parser.parse_args()
set_seed(args.seed)


transform_load = transforms.Compose([
            transforms.ToTensor(),
        ])

dir_v = {'D':args.D_public, 'arch':'resnet18', 'method':'simclr'}
dir_shadow = {'D':args.M_shadow_dataset, 'arch':args.M_shadow_arch, 'method':'simclr'}   # M_shadow
dir_adv = {'D':args.M_suspect_dataset, 'arch':args.M_suspect_arch, 'method':'simclr'}   # M_suspect
n_sample_train = args.n_sample_train  # N_public
n_sample_test = args.n_sample_test   # N_private
n_aug = args.n_aug   # m
n_aug_local = args.n_aug_local  # n
n_epoch = args.n_epoch  # T
lamda = args.lamda  # α,β

def aug_img(img, n, aug_transform):
    aug_imgs = None
    # img = torch.unsqueeze(img, dim=0)
    # print('img:', img.shape)

    for i in range(n):
        aug_img = aug_transform(img)
        # print('aug_img:', aug_img.shape)
        
        if aug_imgs is None:
            aug_imgs = aug_img
        else:
            aug_imgs = torch.cat((aug_imgs, aug_img), dim=0)

    return aug_imgs


if 'cifar10' in dir_v['D']:
    img_size=32
    local_size=32 if dir_v['arch'] == 'vgg16' or dir_shadow['arch'] == 'vgg16' or dir_adv['arch'] == 'vgg16' else 16
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_dataset = CIFAR10(root='D:\Exp\datasets', train=True, transform=transform_load, download=True)
    train_dataset = split_dataset(train_dataset, 10, True)[0]
    test_dataset = CIFAR10(root='D:\Exp\datasets', train=False, transform=transform_load, download=True)
    root_dataset = 'cifar10'
    shadow_dataset = 'cifar100'
    diff_dataset = 'cifar100'
elif 'cifar100' in dir_v['D']:
    img_size=32
    local_size=32 if dir_v['arch'] == 'vgg16' or dir_shadow['arch'] == 'vgg16' or dir_adv['arch'] == 'vgg16' else 16
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_dataset = CIFAR100(root='D:\Exp\datasets', train=True, transform=transform_load, download=True)
    train_dataset = split_dataset(train_dataset, 10, True)[0]
    test_dataset = CIFAR100(root='D:\Exp\datasets', train=False, transform=transform_load, download=True)
    diff_dataset = 'svhn'
elif 'stl10' in dir_v['D']:
    img_size=96
    local_size=64
    mean = (0.507, 0.487, 0.441)
    std = (0.267, 0.256, 0.276)
    train_dataset = STL10(root='D:\Exp\datasets', split='train+unlabeled', transform=transform_load, download=True)
    train_dataset = split_dataset(train_dataset, 10, True)[0]
    test_dataset = STL10(root='D:\Exp\datasets', split='test', transform=transform_load, download=False)
    diff_dataset = 'tiny-imagenet'
elif 'imagenette' in dir_v['D']:
    img_size=224
    local_size=96
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_dataset = ImageNet(root='D:\Exp\datasets/imagenette', train=True)
    train_dataset = split_dataset(train_dataset, 10, True)[0]
    test_dataset = ImageNet(root='D:\Exp\datasets/imagenette', train=False)
    root_dataset = 'imagenette'
    shadow_dataset = 'cifar10'
    diff_dataset = 'imagewoof'
elif 'imagenet' in dir_v['D']:
    img_size=224
    local_size=96
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_dataset = ImageNet(root='D:\Exp\datasets/imagenet', train=True)
    # train_dataset = split_dataset(train_dataset, 10, True)[0]
    root_dataset = 'cifar10'
    test_dataset = ImageNet(root='D:\Exp\datasets/imagenet', train=False)
    shadow_dataset = 'cifar10'
print('Dv:', dir_v['D'])

normalize = transforms.Normalize(mean,std)

# weightPath_v = f'Experiments\{root_dataset}\{dir_v["D"]}-{dir_v["method"]}-{dir_v["arch"]}-clean-True\epoch_800.pth.tar'
# weightPath_v = r'Experiments\imagenet\simclr_resnet50_16xb256-coslr-800e_in1k_20220825-85fcc4de.pth'
# model_fun, feat_dim = model_dict_cifar[dir_v['arch']] if 'cifar' in dir_v['D'] else model_dict[dir_v['arch']]
# if 'dino' in weightPath_v:
#     model_v = model_fun(img_size=[img_size], patch_size=4)
#     utils.load_pretrained_weights(model_v, weightPath_v, 'teacher', 'vit_tiny', patch_size=4)
# else:
#     if 'mae' in weightPath_v:
#         model_v = model_fun(img_size=img_size, global_pool=False)        
#     elif 'moco' in weightPath_v:
#         model_v = model_fun(full_model=True, num_classes=128, mocov3=True)
#         # model_v = MoCoV3ViT(arch='mocov3-small', patch_size=16, norm_eval=True)
#     else:
#         model_v = model_fun()
#         # model_v = ResNet(50, zero_init_residual=False)
#     model_v = load_model(model_v, weightPath_v)
# model_v.eval()
# model_v = model_v.cuda()

# weightPath_shadow = f'D:\Exp\Backdoor\CTRL\CTRL-master\Experiments\{shadow_dataset}\{dir_shadow["D"]}-{dir_shadow["method"]}-{dir_shadow["arch"]}-clean-True\epoch_800.pth.tar'
# weightPath_shadow = r'Experiments\imagenet\mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k-224_20220826-08bc52f7.pth'
weightPath_shadow = args.M_shadow_path
model_fun, feat_dim = model_dict_cifar[dir_shadow['arch']] if 'cifar' in dir_shadow['D'] else model_dict[dir_shadow['arch']]
if 'dino' in weightPath_shadow:
    model_shadow = model_fun(img_size=[img_size], patch_size=16)
    utils.load_pretrained_weights(model_shadow, weightPath_shadow, 'teacher', 'vit_tiny', patch_size=4)
else:
    if 'mae' in weightPath_shadow:
        model_shadow = model_fun(img_size=img_size, global_pool=False)        
    elif 'moco' in weightPath_shadow:
        if dir_shadow['arch'] == 'resnet18':
            model_shadow = model_fun(full_model=True, num_classes=128, mocov3=True)
        elif dir_shadow['arch'] == 'vgg16':
            model_shadow = model_fun(full_model=True, num_classes=128, mocov3=True)
        elif dir_shadow['arch'] == 'resnet50':
            model_shadow = model_fun(full_model=True, num_classes=128, mocov3=True, r18=False)
        # model_shadow = MoCoV3ViT(arch='mocov3-small', patch_size=16, norm_eval=True)
    else:
        model_shadow = model_fun()
        # model_shadow = ResNet(50, zero_init_residual=False)
    model_shadow = load_model(model_shadow, weightPath_shadow)
model_shadow.eval()
model_shadow = model_shadow.cuda()

# weightPath_adv = rf'D:\Exp\Backdoor\CTRL\CTRL-master\Experiments\{root_dataset}\{dir_adv["D"]}-{dir_adv["method"]}-{dir_adv["arch"]}-clean-True\checkpoint0800.pth' if dir_adv["method"]=='dino' else \
#                  rf'D:\Exp\Backdoor\CTRL\CTRL-master\Experiments\{root_dataset}\{dir_adv["D"]}-{dir_adv["method"]}-{dir_adv["arch"]}-clean-True\epoch_800.pth.tar'
# weightPath_adv = r'Experiments\imagenette\imagewoof-simsiam-vgg16-clean-True\epoch_800.pth.tar'
weightPath_adv = args.M_suspect_path
model_fun, feat_dim = model_dict_cifar[dir_adv['arch']] if 'cifar' in dir_adv['D'] else model_dict[dir_adv['arch']]
if 'dino' in weightPath_adv:
    model_adv = model_fun(img_size=[img_size], patch_size=4)
    utils.load_pretrained_weights(model_adv, weightPath_adv, 'teacher', 'vit_tiny', patch_size=4)
else:
    if 'mae' in weightPath_adv:
        model_adv = model_fun(img_size=img_size, global_pool=False)        
    elif 'moco' in weightPath_adv:
        if dir_adv['arch'] == 'resnet18':
            model_adv = model_fun(full_model=True, num_classes=128, mocov3=True)
        elif dir_adv['arch'] == 'vgg16':
            model_adv = model_fun(full_model=True, num_classes=128, mocov3=True)
        elif dir_adv['arch'] == 'resnet50':
            model_adv = model_fun(full_model=True, num_classes=128, mocov3=True, r18=False)
        # model_adv = MoCoV3ViT(arch='base', patch_size=16, norm_eval=True)
    else:
        model_adv = model_fun()
        # model_adv = ResNet(50, zero_init_residual=False)
    model_adv = load_model(model_adv, weightPath_adv)
    # checkpoint = torch.load(weightPath_adv, map_location="cuda")
    # msg = model_adv.load_state_dict(checkpoint, strict=False)
    # print('msg:', msg)
    # print("=> loaded pre-trained model '{}'".format(weightPath_adv))
model_adv.eval()
model_adv = model_adv.cuda()

aug_transform = transforms.Compose([
                                aug.RandomResizedCrop(size = (img_size, img_size), scale=(0.2, 1.0)),
                                aug.RandomHorizontalFlip(),
                                RandomApply(aug.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
                                aug.RandomGrayscale(p=0.2),
                                # RandomPatchMasking(patch_size=4, min_mask_ratio=0.25, max_mask_ratio=0.75, mask_value=0, random=False),
                                normalize,
                            ])  # 就算设置了seed，aug.ColorJitter每次的增强效果也不一样

aug_transform_dino_global = transforms.Compose([
                                aug.RandomResizedCrop(size = (img_size, img_size), scale=(0.4, 1.0)),
                                aug.RandomHorizontalFlip(),
                                RandomApply(aug.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
                                aug.RandomGrayscale(p=0.2, keepdim=True),
                                # RandomApply(GaussianBlur(), p=0.5),
                                # RandomApply(Solarize(), p=0.3),
                                # RandomPatchMasking(patch_size=16, min_mask_ratio=0.25, max_mask_ratio=0.75, mask_value=0, random=True),
                                normalize,
                            ])

aug_transform_dino_local = transforms.Compose([
                                aug.RandomResizedCrop(size = (local_size, local_size), scale=(0.05, 0.4)),
                                aug.RandomHorizontalFlip(),
                                RandomApply(aug.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
                                aug.RandomGrayscale(p=0.2, keepdim=True),
                                # RandomApply(GaussianBlur(), p=0.5),
                                # RandomApply(Solarize(), p=0.3),
                                # RandomPatchMasking(patch_size=16, min_mask_ratio=0.25, max_mask_ratio=0.75, mask_value=0, random=True),
                                normalize,
                            ])  # 就算设置了seed，aug.ColorJitter每次的增强效果也不一样

aug_transform_mocov3 = transforms.Compose([
                                aug.RandomResizedCrop(size=(img_size, img_size), scale=(0.2, 1.)),
                                RandomApply(aug.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
                                aug.RandomGrayscale(p=0.2),
                                RandomApply(GaussianBlur(), p=0.5),
                                aug.RandomHorizontalFlip(),
                                normalize
                            ])

def cut_img(img, n, patch_size=4):
    cut_imgs = None
    
    h_num = (img.size(2)-2*patch_size)/patch_size
    w_num = (img.size(3)-2*patch_size)/patch_size

    for i in range(n):
        rand_h_id = random.randint(1, h_num)
        rand_w_id = random.randint(1, w_num)
        rand_h_id_2 = random.randint(1, h_num)
        rand_w_id_2 = random.randint(1, w_num)

        cut_img_center = img[:,:,rand_h_id*patch_size:(rand_h_id+1)*patch_size,rand_w_id*patch_size:(rand_w_id+1)*patch_size]
        # cut_img_left = img[:,:,rand_h_id*patch_size:(rand_h_id+1)*patch_size,(rand_w_id-1)*patch_size:rand_w_id*patch_size]
        # cut_img_right = img[:,:,rand_h_id*patch_size:(rand_h_id+1)*patch_size,(rand_w_id+1)*patch_size:(rand_w_id+2)*patch_size]
        # cut_img_up = img[:,:,(rand_h_id-1)*patch_size:rand_h_id*patch_size,rand_w_id*patch_size:(rand_w_id+1)*patch_size]
        # cut_img_down = img[:,:,(rand_h_id+1)*patch_size:(rand_h_id+2)*patch_size,rand_w_id*patch_size:(rand_w_id+1)*patch_size]
        cut_img_center_2 = img[:,:,rand_h_id_2*patch_size:(rand_h_id_2+1)*patch_size,rand_w_id_2*patch_size:(rand_w_id_2+1)*patch_size]

        cut_img_center = F.interpolate(cut_img_center, size=(img.size(2), img.size(3)), mode='bilinear', align_corners=True)
        # cut_img_left = F.interpolate(cut_img_left, size=(img.size(2), img.size(3)), mode='bilinear', align_corners=True)
        # cut_img_right = F.interpolate(cut_img_right, size=(img.size(2), img.size(3)), mode='bilinear', align_corners=True)
        # cut_img_up = F.interpolate(cut_img_up, size=(img.size(2), img.size(3)), mode='bilinear', align_corners=True)
        # cut_img_down = F.interpolate(cut_img_down, size=(img.size(2), img.size(3)), mode='bilinear', align_corners=True)
        cut_img_center_2 = F.interpolate(cut_img_center_2, size=(img.size(2), img.size(3)), mode='bilinear', align_corners=True)

        cut_img_center = torch.unsqueeze(cut_img_center, dim=1)
        # cut_img_left = torch.unsqueeze(cut_img_left, dim=1)
        # cut_img_right = torch.unsqueeze(cut_img_right, dim=1)
        # cut_img_up = torch.unsqueeze(cut_img_up, dim=1)
        # cut_img_down = torch.unsqueeze(cut_img_down, dim=1)
        cut_img_center_2 = torch.unsqueeze(cut_img_center_2, dim=1)

        cut_img_center = normalize(cut_img_center)
        # cut_img_left = normalize(cut_img_left)
        # cut_img_right = normalize(cut_img_right)
        # cut_img_up = normalize(cut_img_up)
        # cut_img_down = normalize(cut_img_down)
        cut_img_center_2 = normalize(cut_img_center_2)

        # cut_img = torch.cat((cut_img_center, cut_img_left, cut_img_right, cut_img_up, cut_img_down), dim=1)  # (b,5,c,h,w)
        cut_img = torch.cat((cut_img_center, cut_img_center_2), dim=1)  # (b,2,c,h,w)
        
        if cut_imgs is None:
            cut_imgs = cut_img
        else:
            cut_imgs = torch.cat((cut_imgs, cut_img), dim=0)

    return cut_imgs

def getSim(weightPath_v, model_v, rand_seed):
    res_sim_gg = None
    res_sim_ll = None
    res_sim_gl = None
    res_sim = None

    set_seed(rand_seed)
    pbar = tqdm(total=n_epoch, desc='Processing')
    for i in range(n_epoch):
        indices = np.random.choice(len(train_dataset), n_sample_train, replace=False)
        train_loader = DataLoader(Subset(train_dataset, indices), batch_size=1, shuffle=True)
        # print(indices[:10])
        indices = np.random.choice(len(test_dataset), n_sample_test, replace=False)
        test_loader = DataLoader(Subset(test_dataset, indices), batch_size=1, shuffle=True)

        sim_v_gg = None
        sim_v_gg_test = None
        sim_v_ll = None
        sim_v_ll_test = None
        sim_v_gl = None
        sim_v_gl_test = None
        sim_v = None
        sim_v_test = None

        # 求多个样本的多个增强之间的cos_mat（也是边的权重）相似度
        outputs_v = None
        outputs_v_local = None
        edges_v = None
        edges_v_local = None
        edges_sim_v_gg = None
        edges_sim_v_ll = None
        edges_sim_v_gl = None

        # pbar = tqdm(total=len(train_loader), desc='Processing')
        for data in train_loader:
            imgs_tensor, target, index = data
            # print(imgs_tensor.shape)

            aug_imgs = aug_img(imgs_tensor, n_aug, aug_transform_dino_global)
            aug_imgs = aug_imgs.cuda()
            
            aug_imgs_local = aug_img(imgs_tensor, n_aug_local, aug_transform_dino_local)
            aug_imgs_local = aug_imgs_local.cuda()

            if 'mocov3_vit' not in weightPath_v:
                output_v_aug = model_v(aug_imgs) if 'mae' not in weightPath_v else model_v.forward_features(aug_imgs)  # (n_aug,d)
            else:
                output_v_aug = model_v(aug_imgs)[0][1] if 'mae' not in weightPath_v else model_v.forward_features(aug_imgs)
            
            output_v_aug = output_v_aug.detach().cpu()

            if 'mocov3_vit' not in weightPath_v:
                output_v_aug_local = model_v(aug_imgs_local) if 'mae' not in weightPath_v else model_v.forward_features(aug_imgs_local)  # (n_aug_local,d)
            else:
                output_v_aug_local = model_v(aug_imgs_local)[0][1] if 'mae' not in weightPath_v else model_v.forward_features(aug_imgs_local)
            # 
            output_v_aug_local = output_v_aug_local.detach().cpu()

            # 求单样本的多个增强之间的两两相似度
            # global与local相似度
            # mean_cos_v_gl = torch.mean(F.cosine_similarity(output_v_aug, output_v_aug_local))
            mean_cos_v_gl = None
            for i in range(output_v_aug.size(0)):
                for j in range(output_v_aug_local.size(0)):
                    cos = F.cosine_similarity(output_v_aug[i], output_v_aug_local[j], dim=0)
                    cos = torch.unsqueeze(cos, dim=0)
                    if mean_cos_v_gl is None:
                        mean_cos_v_gl = cos
                    else:
                        mean_cos_v_gl = torch.cat((mean_cos_v_gl, cos), dim=0)
            mean_cos_v_gl = torch.mean(mean_cos_v_gl, dim=0)  # (1,)

            mat_cos_v_gg, mean_cos_v_gg = cal_Cosinesimilarity(output_v_aug)  # (1,)

            mat_cos_v_ll, mean_cos_v_ll = cal_Cosinesimilarity(output_v_aug_local)  # (1,)

            mean_cos_v_gl = torch.unsqueeze(mean_cos_v_gl, dim=0)
            mean_cos_v_gg = torch.unsqueeze(mean_cos_v_gg, dim=0)
            mean_cos_v_ll = torch.unsqueeze(mean_cos_v_ll, dim=0)

            # print('mean_cos_v_gl:', mean_cos_v_gl.shape)

            # lambda_s_v_gl = 1
            # lambda_s_v_gg = 1
            # lambda_s_v_ll = 1

            # mean_cos_v = lambda_s_v_gl*mean_cos_v_gl + lambda_s_v_gg*mean_cos_v_gg + lambda_s_v_ll*mean_cos_v_ll  # (1,)
            mean_cos_v = torch.cat((mean_cos_v_gg*1 if mean_cos_v_gg>0 else mean_cos_v_gg,
                                    mean_cos_v_ll*1 if mean_cos_v_ll>0 else mean_cos_v_ll,
                                    mean_cos_v_gl*1 if mean_cos_v_gl>0 else mean_cos_v_gl), dim=0)

            mean_cos_v_gg = torch.unsqueeze(mean_cos_v_gg, dim=0)
            mean_cos_v_ll = torch.unsqueeze(mean_cos_v_ll, dim=0)
            mean_cos_v_gl = torch.unsqueeze(mean_cos_v_gl, dim=0)
            mean_cos_v = torch.unsqueeze(mean_cos_v, dim=0)

            if sim_v is None:
                sim_v_gg = mean_cos_v_gg
                sim_v_ll = mean_cos_v_ll
                sim_v_gl = mean_cos_v_gl
                sim_v = mean_cos_v
            else:
                sim_v_gg = torch.cat((sim_v_gg, mean_cos_v_gg), dim=0)  # (n_sample_train, 1)
                sim_v_ll = torch.cat((sim_v_ll, mean_cos_v_ll), dim=0)  # (n_sample_train, 1)
                sim_v_gl = torch.cat((sim_v_gl, mean_cos_v_gl), dim=0)  # (n_sample_train, 1)
                sim_v = torch.cat((sim_v, mean_cos_v), dim=0)  # (n_sample_train, 1)

            del mean_cos_v_gg
            del mean_cos_v_ll
            del mean_cos_v_gl
            del mat_cos_v_gg
            del mat_cos_v_ll
            del mean_cos_v

            #---------------------------------------------------

            output_v_aug = torch.unsqueeze(output_v_aug, dim=1)
            output_v_aug_local = torch.unsqueeze(output_v_aug_local, dim=1)

            if outputs_v is None:
                outputs_v = output_v_aug
                outputs_v_local = output_v_aug_local
            else:
                outputs_v = torch.cat((outputs_v, output_v_aug), dim=1)
                outputs_v_local = torch.cat((outputs_v_local, output_v_aug_local), dim=1)
            
            del output_v_aug
            del output_v_aug_local

        #     pbar.update()
        # pbar.close()

        for i in range(outputs_v.size(0)):
            mat_cos_v, mean_cos_v = cal_Cosinesimilarity(outputs_v[i])
            mat_cos_v = torch.unsqueeze(mat_cos_v, dim=0)

            if edges_v is None:
                edges_v = mat_cos_v
            else:
                edges_v = torch.cat((edges_v, mat_cos_v), dim=0)
        
        for i in range(outputs_v_local.size(0)):
            mat_cos_v_local, mean_cos_v_local = cal_Cosinesimilarity(outputs_v_local[i])
            mat_cos_v_local = torch.unsqueeze(mat_cos_v_local, dim=0)

            if edges_v_local is None:
                edges_v_local = mat_cos_v_local
            else:
                edges_v_local = torch.cat((edges_v_local, mat_cos_v_local), dim=0)

        for i in range(len(edges_v)):
            for j in range(i+1, len(edges_v)):
                edge_sim_v_gg = torch.sum(torch.abs(edges_v[i]-edges_v[j]))/((edges_v[i].size(0)-1)*edges_v[i].size(0))
                edge_sim_v_gg = torch.unsqueeze(edge_sim_v_gg, dim=0)
                
                if edges_sim_v_gg is None:
                    edges_sim_v_gg = edge_sim_v_gg
                else:
                    edges_sim_v_gg = torch.cat((edges_sim_v_gg, edge_sim_v_gg), dim=0)  #((n_aug-1)!, )
        edges_sim_v_gg = torch.unsqueeze(torch.mean(edges_sim_v_gg, dim=0), dim=0)
        
        for i in range(len(edges_v_local)):
            for j in range(i+1, len(edges_v_local)):
                edge_sim_v_ll = torch.sum(torch.abs(edges_v_local[i]-edges_v_local[j]))/((edges_v_local[i].size(0)-1)*edges_v_local[i].size(0))
                edge_sim_v_ll = torch.unsqueeze(edge_sim_v_ll, dim=0)
                
                if edges_sim_v_ll is None:
                    edges_sim_v_ll = edge_sim_v_ll
                else:
                    edges_sim_v_ll = torch.cat((edges_sim_v_ll, edge_sim_v_ll), dim=0)  #((n_aug_local-1)!, )
        edges_sim_v_ll = torch.unsqueeze(torch.mean(edges_sim_v_ll, dim=0), dim=0)

        for i in range(len(edges_v)):
            for j in range(len(edges_v_local)):
                edge_sim_v_gl = torch.sum(torch.abs(edges_v[i]-edges_v_local[j]))/((edges_v[i].size(0)-1)*edges_v[i].size(0))
                edge_sim_v_gl = torch.unsqueeze(edge_sim_v_gl, dim=0)

                if edges_sim_v_gl is None:
                    edges_sim_v_gl = edge_sim_v_gl
                else:
                    edges_sim_v_gl = torch.cat((edges_sim_v_gl, edge_sim_v_gl), dim=0)  #(n_aug*n_aug_local, )
        edges_sim_v_gl = torch.unsqueeze(torch.mean(edges_sim_v_gl, dim=0), dim=0)

        outputs_v_test = None
        outputs_v_local_test = None
        edges_v_test = None
        edges_v_local_test = None
        edges_sim_v_gg_test = None
        edges_sim_v_ll_test = None
        edges_sim_v_gl_test = None

        # pbar = tqdm(total=len(test_loader), desc='Processing')
        for data in test_loader:
            imgs_tensor, target, index = data
            
            aug_imgs = aug_img(imgs_tensor, n_aug, aug_transform_dino_global)
            aug_imgs = aug_imgs.cuda()
            
            aug_imgs_local = aug_img(imgs_tensor, n_aug_local, aug_transform_dino_local)
            aug_imgs_local = aug_imgs_local.cuda()

            if 'mocov3_vit' not in weightPath_v:
                output_v_aug = model_v(aug_imgs) if 'mae' not in weightPath_v else model_v.forward_features(aug_imgs)  # (n_aug,d)
            else:
                output_v_aug = model_v(aug_imgs)[0][1] if 'mae' not in weightPath_v else model_v.forward_features(aug_imgs)
            
            output_v_aug = output_v_aug.detach().cpu()

            if 'mocov3_vit' not in weightPath_v:
                output_v_aug_local = model_v(aug_imgs_local) if 'mae' not in weightPath_v else model_v.forward_features(aug_imgs_local)  # (n_aug_local,d)
            else:
                output_v_aug_local = model_v(aug_imgs_local)[0][1] if 'mae' not in weightPath_v else model_v.forward_features(aug_imgs_local)

            output_v_aug_local = output_v_aug_local.detach().cpu()
            

            mean_cos_v_gl = None
            for i in range(output_v_aug.size(0)):
                for j in range(output_v_aug_local.size(0)):
                    cos = F.cosine_similarity(output_v_aug[i], output_v_aug_local[j], dim=0)
                    cos = torch.unsqueeze(cos, dim=0)
                    if mean_cos_v_gl is None:
                        mean_cos_v_gl = cos
                    else:
                        mean_cos_v_gl = torch.cat((mean_cos_v_gl, cos), dim=0)
            mean_cos_v_gl = torch.mean(mean_cos_v_gl, dim=0)
            
            mat_cos_v_gg, mean_cos_v_gg = cal_Cosinesimilarity(output_v_aug)
            
            mat_cos_v_ll, mean_cos_v_ll = cal_Cosinesimilarity(output_v_aug_local)

            mean_cos_v_gl = torch.unsqueeze(mean_cos_v_gl, dim=0)
            mean_cos_v_gg = torch.unsqueeze(mean_cos_v_gg, dim=0)
            mean_cos_v_ll = torch.unsqueeze(mean_cos_v_ll, dim=0)

            # lambda_s_v_gl = 1
            # lambda_s_v_gg = 1
            # lambda_s_v_ll = 1

            # mean_cos_v = lambda_s_v_gl*mean_cos_v_gl + lambda_s_v_gg*mean_cos_v_gg + lambda_s_v_ll*mean_cos_v_ll
            mean_cos_v = torch.cat((mean_cos_v_gg*1 if mean_cos_v_gg>0 else mean_cos_v_gg,
                                    mean_cos_v_ll*1 if mean_cos_v_ll>0 else mean_cos_v_ll,
                                    mean_cos_v_gl*1 if mean_cos_v_gl>0 else mean_cos_v_gl), dim=0)

            mean_cos_v_gg = torch.unsqueeze(mean_cos_v_gg, dim=0)
            mean_cos_v_ll = torch.unsqueeze(mean_cos_v_ll, dim=0)
            mean_cos_v_gl = torch.unsqueeze(mean_cos_v_gl, dim=0)
            mean_cos_v = torch.unsqueeze(mean_cos_v, dim=0)

            if sim_v_test is None:
                sim_v_gg_test = mean_cos_v_gg
                sim_v_ll_test = mean_cos_v_ll
                sim_v_gl_test = mean_cos_v_gl
                sim_v_test = mean_cos_v
            else:
                sim_v_gg_test = torch.cat((sim_v_gg_test, mean_cos_v_gg), dim=0)
                sim_v_ll_test = torch.cat((sim_v_ll_test, mean_cos_v_ll), dim=0)
                sim_v_gl_test = torch.cat((sim_v_gl_test, mean_cos_v_gl), dim=0)
                sim_v_test = torch.cat((sim_v_test, mean_cos_v), dim=0)
            
            del mean_cos_v_gg
            del mean_cos_v_ll
            del mean_cos_v_gl
            del mat_cos_v_gg
            del mat_cos_v_ll
            del mean_cos_v

            #---------------------------------------------------

            output_v_aug = torch.unsqueeze(output_v_aug, dim=1)
            output_v_aug_local = torch.unsqueeze(output_v_aug_local, dim=1)

            if outputs_v_test is None:
                outputs_v_test = output_v_aug
                outputs_v_local_test = output_v_aug_local
            else:
                outputs_v_test = torch.cat((outputs_v_test, output_v_aug), dim=1)
                outputs_v_local_test = torch.cat((outputs_v_local_test, output_v_aug_local), dim=1)

            del output_v_aug
            del output_v_aug_local


        for i in range(outputs_v_test.size(0)):
            mat_cos_v, mean_cos_v = cal_Cosinesimilarity(outputs_v_test[i])
            mat_cos_v = torch.unsqueeze(mat_cos_v, dim=0)

            if edges_v_test is None:
                edges_v_test = mat_cos_v
            else:
                edges_v_test = torch.cat((edges_v_test, mat_cos_v), dim=0)

        for i in range(outputs_v_local_test.size(0)):
            mat_cos_v_local, mean_cos_v_local = cal_Cosinesimilarity(outputs_v_local_test[i])
            mat_cos_v_local = torch.unsqueeze(mat_cos_v_local, dim=0)

            if edges_v_local_test is None:
                edges_v_local_test = mat_cos_v_local
            else:
                edges_v_local_test = torch.cat((edges_v_local_test, mat_cos_v_local), dim=0)

        for i in range(len(edges_v_test)):
            for j in range(i+1, len(edges_v_test)):
                # global与global相似度
                edge_sim_v_gg_test = torch.sum(torch.abs(edges_v_test[i]-edges_v_test[j]))/((edges_v_test[i].size(0)-1)*edges_v_test[i].size(0))
                edge_sim_v_gg_test = torch.unsqueeze(edge_sim_v_gg_test, dim=0)        

                if edges_sim_v_gg_test is None:
                    edges_sim_v_gg_test = edge_sim_v_gg_test
                else:
                    edges_sim_v_gg_test = torch.cat((edges_sim_v_gg_test, edge_sim_v_gg_test), dim=0)
        edges_sim_v_gg_test = torch.unsqueeze(torch.mean(edges_sim_v_gg_test, dim=0), dim=0)
        
        for i in range(len(edges_v_local_test)):
            for j in range(i+1, len(edges_v_local_test)):
                edge_sim_v_ll_test = torch.sum(torch.abs(edges_v_local_test[i]-edges_v_local_test[j]))/((edges_v_local_test[i].size(0)-1)*edges_v_local_test[i].size(0))
                edge_sim_v_ll_test = torch.unsqueeze(edge_sim_v_ll_test, dim=0)
                
                if edges_sim_v_ll_test is None:
                    edges_sim_v_ll_test = edge_sim_v_ll_test
                else:
                    edges_sim_v_ll_test = torch.cat((edges_sim_v_ll_test, edge_sim_v_ll_test), dim=0)
        edges_sim_v_ll_test = torch.unsqueeze(torch.mean(edges_sim_v_ll_test, dim=0), dim=0)

        for i in range(len(edges_v_test)):
            for j in range(len(edges_v_local_test)):
                edge_sim_v_gl_test = torch.sum(torch.abs(edges_v_test[i]-edges_v_local_test[j]))/((edges_v_test[i].size(0)-1)*edges_v_test[i].size(0))
                edge_sim_v_gl_test = torch.unsqueeze(edge_sim_v_gl_test, dim=0)

                if edges_sim_v_gl_test is None:
                    edges_sim_v_gl_test = edge_sim_v_gl_test
                else:
                    edges_sim_v_gl_test = torch.cat((edges_sim_v_gl_test, edge_sim_v_gl_test), dim=0)
        edges_sim_v_gl_test = torch.unsqueeze(torch.mean(edges_sim_v_gl_test, dim=0), dim=0)

        delta_sim_v_gg = torch.mean(sim_v_gg)-torch.mean(sim_v_gg_test)  # (1,)
        delta_sim_v_ll = torch.mean(sim_v_ll)-torch.mean(sim_v_ll_test)  # (1,)
        delta_sim_v_gl = torch.mean(sim_v_gl)-torch.mean(sim_v_gl_test)  # (1,)
        # delta_sim_v = torch.mean(sim_v)-torch.mean(sim_v_test)  # (1,)

        lambda_v_gl = lamda if delta_sim_v_gl > 0 else 1
        lambda_v_gg = lamda if delta_sim_v_gg > 0 else 1
        lambda_v_ll = lamda if delta_sim_v_ll > 0 else 1
        # lambda_v_gl = 1
        # lambda_v_gg = lamda if delta_sim_v_gg > 0 else 1
        # lambda_v_ll = 1

        delta_sim_v = lambda_v_gg*delta_sim_v_gg + lambda_v_ll*delta_sim_v_ll + lambda_v_gl*delta_sim_v_gl
        # print('delta_sim_v:', delta_sim_v)


        delta_sim_v_gg = torch.unsqueeze(delta_sim_v_gg, dim=0)
        delta_sim_v_ll = torch.unsqueeze(delta_sim_v_ll, dim=0)
        delta_sim_v_gl = torch.unsqueeze(delta_sim_v_gl, dim=0)
        delta_sim_v = torch.unsqueeze(delta_sim_v, dim=0)
        # print('delta_sim_v_gg:', delta_sim_v_gg.shape)
        # print('delta_sim_v_ll:', delta_sim_v_ll.shape)
        # print('delta_sim_v_gl:', delta_sim_v_gl.shape)
        # print('delta_sim_v:', delta_sim_v.shape)

        delta_edges_sim_v_gg = edges_sim_v_gg_test-edges_sim_v_gg  #(n_aug*(n_aug-1), )
        delta_edges_sim_v_ll = edges_sim_v_ll_test-edges_sim_v_ll  #(n_aug_local*(n_aug_local-1), )
        delta_edges_sim_v_gl = edges_sim_v_gl_test-edges_sim_v_gl  #(n_aug*(n_aug_local), )

        # delta_edges_sim_v_gg = delta_edges_sim_v_gg.repeat(n_aug_local*(n_aug_local-1))
        # delta_edges_sim_v_ll = delta_edges_sim_v_ll.repeat(n_aug*(n_aug-1))
        # delta_edges_sim_v_gl = delta_edges_sim_v_gl.repeat((n_aug_local-1)*(n_aug-1))

        lambda_d_v_gl = lamda if torch.mean(delta_edges_sim_v_gl) > 0 else 1
        lambda_d_v_gg = lamda if torch.mean(delta_edges_sim_v_gg) > 0 else 1
        lambda_d_v_ll = lamda if torch.mean(delta_edges_sim_v_ll) > 0 else 1
        # lambda_d_v_gl = 1
        # lambda_d_v_gg = lamda if torch.mean(delta_edges_sim_v_gg) > 0 else 1
        # lambda_d_v_ll = 1

        delta_edges_sim_v = lambda_d_v_gg*delta_edges_sim_v_gg + lambda_d_v_ll*delta_edges_sim_v_ll + lambda_d_v_gl*delta_edges_sim_v_gl
        # print('delta_edges_sim_v:', delta_edges_sim_v)
        # delta_edges_sim_v = torch.cat((delta_edges_sim_v_gg, delta_edges_sim_v_ll, delta_edges_sim_v_gl), dim=0)  # (n_aug*n_aug_local+(n_aug_local-1)!+(n_aug-1)!,)
        # print('delta_edges_sim_v_gg:', delta_edges_sim_v_gg.shape)
        # print('delta_edges_sim_v_ll:', delta_edges_sim_v_ll.shape)
        # print('delta_edges_sim_v_gl:', delta_edges_sim_v_gl.shape)
        # print('delta_edges_sim_v:', delta_edges_sim_v.shape)

        # delta_edges_sim_v_gg = torch.unsqueeze(delta_edges_sim_v_gg, dim=0)
        # delta_edges_sim_v_ll = torch.unsqueeze(delta_edges_sim_v_ll, dim=0)
        # delta_edges_sim_v_gl = torch.unsqueeze(delta_edges_sim_v_gl, dim=0)
        # delta_edges_sim_v = torch.unsqueeze(delta_edges_sim_v, dim=0)

        # print('delta_edges_sim_v_gg:', delta_edges_sim_v_gg.shape)
        # print('delta_edges_sim_v_ll:', delta_edges_sim_v_ll.shape)
        # print('delta_edges_sim_v_gl:', delta_edges_sim_v_gl.shape)
        # print('delta_edges_sim_v:', delta_edges_sim_v.shape)

        # delta_sim_v_gg = delta_sim_v_gg.repeat(delta_edges_sim_v_gg.size(0))
        # delta_sim_v_ll = delta_sim_v_ll.repeat(delta_edges_sim_v_ll.size(0))
        # delta_sim_v_gl = delta_sim_v_gl.repeat(delta_edges_sim_v_gl.size(0))

        sim_gg = torch.cat((delta_sim_v_gg, delta_edges_sim_v_gg), dim=0)  # (1+(n_aug-1)!,)
        sim_gl = torch.cat((delta_sim_v_gl, delta_edges_sim_v_gl), dim=0)  # (1+n_aug*n_aug_local,)
        sim_ll = torch.cat((delta_sim_v_ll, delta_edges_sim_v_ll), dim=0)  # (1+(n_aug_local-1)!,)
        sim = torch.cat((delta_sim_v, delta_edges_sim_v), dim=0)  # (1+n_aug*n_aug_local+(n_aug_local-1)!+(n_aug-1)!,)
        # sim = torch.cat((sim_gg, sim_ll, sim_gl), dim=0)
        # print('sim:', sim.shape)
        
        if res_sim is None:
            res_sim_gg = sim_gg
            res_sim_gl = sim_gl
            res_sim_ll = sim_ll
            res_sim = sim
        else:
            res_sim_gg = torch.cat((res_sim_gg, sim_gg), dim=0)  # (n_epoch*(1+(n_aug-1)!),)
            res_sim_gl = torch.cat((res_sim_gl, sim_gl), dim=0)  # (n_epoch*(1+n_aug*n_aug_local),)
            res_sim_ll = torch.cat((res_sim_ll, sim_ll), dim=0)  # (n_epoch*(1+(n_aug_local-1)!),)
            res_sim = torch.cat((res_sim, sim), dim=0)  # (n_epoch*(1+n_aug*n_aug_local+(n_aug_local-1)!+(n_aug-1)!),)

        pbar.update()
    pbar.close()
        
    return np.array(res_sim), np.array(res_sim_gg), np.array(res_sim_gl), np.array(res_sim_ll)

T,P = [],[]
# set_seed(9999)
for i in range(3):
    rand_seed = random.randint(0, 2**32-1)
    # res_sim_v, res_sim_v_gg, res_sim_v_gl, res_sim_v_ll = getSim(weightPath_v, model_v)
    res_sim_shadow, res_sim_shadow_gg, res_sim_shadow_gl, res_sim_shadow_ll = getSim(weightPath_shadow, model_shadow, rand_seed)
    res_sim_adv, res_sim_adv_gg, res_sim_adv_gl, res_sim_adv_ll = getSim(weightPath_adv, model_adv, rand_seed)

    t, p_shadow = one_tailed_ttest(res_sim_adv, res_sim_shadow)
    print('t:', t)
    print('p_shadow:', p_shadow)
    print('delta_S:', np.mean(res_sim_adv-res_sim_shadow))
    print('steal:', p_shadow < 0.05)   # 成立即为偷窃
    print('-------')
    t_gg, p_shadow_gg = one_tailed_ttest(res_sim_adv_gg, res_sim_shadow_gg)
    print('t_gg:', t_gg)
    print('p_shadow_gg:', p_shadow_gg)
    print('delta_S:', np.mean(res_sim_adv_gg-res_sim_shadow_gg))
    print('-------')
    t_gl, p_shadow_gl = one_tailed_ttest(res_sim_adv_gl, res_sim_shadow_gl)
    print('t_gl:', t_gl)
    print('p_shadow_gl:', p_shadow_gl)
    print('delta_S:', np.mean(res_sim_adv_gl-res_sim_shadow_gl))
    print('-------')
    t_ll, p_shadow_ll = one_tailed_ttest(res_sim_adv_ll, res_sim_shadow_ll)
    print('t_ll:', t_ll)
    print('p_shadow_ll:', p_shadow_ll)
    print('delta_S:', np.mean(res_sim_adv_ll-res_sim_shadow_ll))

    T.append(t)
    P.append(p_shadow)
print('t:', np.mean(T))
print('p:', np.mean(P))
print('sig_t:', np.std(T))
print('sig_p:', np.std(P))
print('2sig_t:', 2*np.std(T))
print('2sig_p:', 2*np.std(P))

# res_sim_shadow, res_sim_shadow_gg, res_sim_shadow_gl, res_sim_shadow_ll = getSim(weightPath_shadow, model_shadow, 999)
# res_sim_adv, res_sim_adv_gg, res_sim_adv_gl, res_sim_adv_ll = getSim(weightPath_adv, model_adv, 999)
# # p_v = stats.ttest_ind(res_sim_v, res_sim_adv)[1]/2
# t, p_shadow = one_tailed_ttest(res_sim_adv, res_sim_shadow)
# # p_v = stats.ks_2samp(res_sim_v, res_sim_adv)[1]
# # p_shadow = stats.ks_2samp(res_sim_shadow, res_sim_adv)[1]/2
# # print('p_v:', p_v)
# # print('delta_S1:', np.mean(res_sim_v-res_sim_adv))
# print('t:', t)
# print('p_shadow:', p_shadow)
# print('delta_S:', np.mean(res_sim_adv-res_sim_shadow))
# # print('not steal:', p_v < 0.05)   # 成立即为未偷窃
# print('steal:', p_shadow < 0.05)   # 成立即为偷窃
# # print(p_v>p_shadow*10)
# print('-------')
# # p_v_gg = stats.ttest_ind(res_sim_v_gg, res_sim_adv_gg)[1]/2
# t_gg, p_shadow_gg = one_tailed_ttest(res_sim_adv_gg, res_sim_shadow_gg)
# # p_v_gg = stats.ks_2samp(res_sim_v_gg, res_sim_adv_gg)[1]
# # p_shadow_gg = stats.ks_2samp(res_sim_shadow_gg, res_sim_adv_gg)[1]/2
# # print('p_v_gg:', p_v_gg)
# print('t_gg:', t_gg)
# print('p_shadow_gg:', p_shadow_gg)
# print('delta_S:', np.mean(res_sim_adv_gg-res_sim_shadow_gg))
# # print(p_shadow_gg < 0.05)   
# # print(p_v_gg>p_shadow_gg*10)
# print('-------')
# # p_v_gl = stats.ttest_ind(res_sim_v_gl, res_sim_adv_gl)[1]/2
# t_gl, p_shadow_gl = one_tailed_ttest(res_sim_adv_gl, res_sim_shadow_gl)
# # p_v_gl = stats.ks_2samp(res_sim_v_gl, res_sim_adv_gl)[1]
# # p_shadow_gl = stats.ks_2samp(res_sim_shadow_gl, res_sim_adv_gl)[1]/2
# # print('p_v_gl:', p_v_gl)
# print('t_gl:', t_gl)
# print('p_shadow_gl:', p_shadow_gl)
# print('delta_S:', np.mean(res_sim_adv_gl-res_sim_shadow_gl))
# # print(p_shadow_gl < 0.05)   
# # print(p_v_gl>p_shadow_gl*10)
# print('-------')
# # p_v_ll = stats.ttest_ind(res_sim_v_ll, res_sim_adv_ll)[1]/2
# t_ll, p_shadow_ll = one_tailed_ttest(res_sim_adv_ll, res_sim_shadow_ll)
# # p_v_ll = stats.ks_2samp(res_sim_v_ll, res_sim_adv_ll)[1]
# # p_shadow_ll = stats.ks_2samp(res_sim_shadow_ll, res_sim_adv_ll)[1]/2
# # print('p_v_ll:', p_v_ll)
# print('t_ll:', t_ll)
# print('p_shadow_ll:', p_shadow_ll)
# print('delta_S:', np.mean(res_sim_adv_ll-res_sim_shadow_ll))
# # print(p_shadow_ll < 0.05)   
# # print(p_v_ll>p_shadow_ll*10)


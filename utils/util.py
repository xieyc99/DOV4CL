import os 
import random
import numpy as np
import torch
from PIL import Image
import torchvision
import time
import sys
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torch.nn as nn
from utils.pos_embed import interpolate_pos_embed
import scipy.stats as stats

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # print(pred)
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))
        # print(correct)

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def set_seed(seed):
    # for reproducibility. 
    # note that pytorch is not completely reproducible 
    # https://pytorch.org/docs/stable/notes/randomness.html  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.initial_seed() # dataloader multi processing 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    return None

def save_model(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_model(model, path):
    # model_path = os.path.join(path, 'epoch_301.pth.tar')
    model_path = path
    checkpoint = torch.load(model_path, map_location="cuda")
    # print(checkpoint.keys())

    if 'mae' in model_path:
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        # for name, param in model.named_parameters():
        #     print(f"Layer: {name}, Shape: {param.shape}")

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed(model, checkpoint_model)
       
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print('msg:', msg)
        # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

        print("=> loaded pre-trained model '{}'".format(model_path))
    else:
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        # state_dict = checkpoint

        # for k in list(state_dict.keys()):
        #     print(k)
        
        new_state_dict = {}
        for k in list(state_dict.keys()):
            if 'moco' in path:
                if 'encoder_q' in k and 'distill' not in k:
                    if k.startswith('module.'):
                        new_state_dict[k[len("module.")+ len("encoder_q."):]] = state_dict[k]
                    else:
                        if 'backbone_k' not in k:
                            new_state_dict[k[len("encoder_q."):]] = state_dict[k]
                # if 'backbone' in k and 'distill' not in k:  # for imagenet pre-train model
                #     if k.startswith('module.'):
                #         new_state_dict[k[len("module.")+ len("backbone."):]] = state_dict[k]
                #     else:
                #         if 'backbone_k' not in k:
                #             new_state_dict[k[len("backbone."):]] = state_dict[k]
            else:
                if 'backbone' in k and 'distill' not in k:
                    if k.startswith('module.'):
                        new_state_dict[k[len("module.")+ len("backbone."):]] = state_dict[k]
                    else:
                        if 'backbone_k' not in k:
                            new_state_dict[k[len("backbone."):]] = state_dict[k]
        # print(new_state_dict)
        msg = model.load_state_dict(new_state_dict, strict=False)
        print('msg:', msg)

        print("=> loaded pre-trained model '{}'".format(model_path))
        # print(checkpoint['epoch'])
    
    return model

def cal_Cosinesimilarity(tensor):
    norms = torch.norm(tensor, dim=1, keepdim=True)

    tensor_transposed = tensor.t()

    dot_products = torch.matmul(tensor, tensor_transposed)

    cosine_similarity = dot_products / (norms * norms.t())

    cosine_similarity = torch.clamp(cosine_similarity, 0, 1)

    k = tensor.size(0)
    n = k*(k-1)/2
    s = (torch.sum(cosine_similarity)-k)/2
    mean_cos = s/n

    return cosine_similarity, mean_cos

def pca_tensor(input, k):
    data_mean = torch.mean(input, dim=0)
    data_std = torch.std(input, dim=0)
    data_normalized = torch.zeros_like(input)
    for i in range(data_normalized.shape[0]):
        if data_std[i] != 0:
            data_normalized[i] = (input[i] - data_mean[i]) / data_std[i]

    pca = PCA(n_components=k)
    data_normalized = pca.fit_transform(data_normalized.numpy())
    data_normalized = torch.tensor(data_normalized, dtype=torch.float)
    
    return data_normalized

def create_edge_index(n):
    # Creating the first row
    row1 = torch.arange(n, dtype=torch.long).repeat_interleave(n)
    # Creating the second row
    row2 = torch.arange(n, dtype=torch.long).repeat(n)
    # Combining the two rows
    tensor = torch.stack([row1, row2])
    # tensor = torch.tensor(tensor, dtype=torch.long)
    return tensor

class RandomPatchMasking(nn.Module):
    def __init__(self, patch_size=16, min_mask_ratio=0.25, max_mask_ratio=0.75, mask_value=0, random=True):

        super().__init__()
        self.patch_size = patch_size
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.mask_value = mask_value
        self.random = random

    def forward(self, x):

        _, _, height, width = x.shape
        num_patches_along_height = height // self.patch_size
        num_patches_along_width = width // self.patch_size

        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # print(patches.shape)
        patches = patches.contiguous().view(*patches.shape[:4], -1)
        # print(patches.shape)

        total_patches = num_patches_along_height * num_patches_along_width
        if self.random:
            rand_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)
            num_mask_patches = int(rand_ratio * total_patches)
        else:
            num_mask_patches = int(self.max_mask_ratio * total_patches)

        idxs = torch.randperm(total_patches, device=x.device)[:num_mask_patches]
        batch_size, num_channels, _, _, patch_size_squared = patches.shape
        patches.view(batch_size, num_channels, total_patches, patch_size_squared)[:, :, idxs, :] = self.mask_value

        reconstructed = patches.view(batch_size, num_channels, num_patches_along_height, num_patches_along_width, self.patch_size, self.patch_size)
        reconstructed = reconstructed.permute(0, 1, 2, 4, 3, 5).contiguous().view(batch_size, num_channels, height, width)

        return reconstructed

    def __call__(self, x):
        return self.forward(x)

def one_tailed_ttest(res_sim_adv, res_sim_shadow):
    t_stat, p_val = stats.ttest_ind(res_sim_adv, res_sim_shadow)
    # t_stat, p_val = stats.ttest_rel(res_sim_adv, res_sim_shadow)

    if t_stat > 0:
        p_val_one_sided = p_val / 2
    else:
        p_val_one_sided = 1 - p_val / 2
    
    return t_stat, p_val_one_sided
    
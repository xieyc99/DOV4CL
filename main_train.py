import os
import sys 
import argparse
import warnings 

from utils.frequency import PoisonFre

import torch.optim as optim
import torch.backends.cudnn as cudnn 


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from methods import set_model
from methods.base import CLTrainer
from utils.util import *
from loaders.diffaugment import set_aug_diff, PoisonAgent
import wandb

parser = argparse.ArgumentParser(description='CTRL Training')

### dataloader
parser.add_argument('--data_path', default='D:\exp\datasets')
parser.add_argument('--steal_dataset', default='cifar10', choices=['cifar10', 'imagenette'])
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'svhn', 'imagewoof', 'imagenette'])
parser.add_argument('--part', default='before', choices=['before', 'after', ''])
parser.add_argument('--unlabel', default=True)
parser.add_argument('--disable_normalize', action='store_true', default=False)
parser.add_argument('--full_dataset', action='store_true', default=True)
parser.add_argument('--window_size', default = 32, type=int)
parser.add_argument('--eval_batch_size', default = 512, type=int)
parser.add_argument('--num_workers', default=0, type=int)


### training
parser.add_argument('--arch', default='resnet18', type=str, choices=['vgg16', 'resnet18'])
parser.add_argument('--method', default = 'simclr', choices=['simclr', 'byol', 'simsiam', 'mocov3'])
parser.add_argument('--batch_size', default = 512, type=int)
parser.add_argument('--epochs', default = 800, type=int)
parser.add_argument('--start_epoch', default = 0, type=int)
parser.add_argument('--remove', default = 'none', choices=['crop', 'flip', 'color', 'gray', 'none'])
parser.add_argument('--poisoning', action='store_true', default=False)
parser.add_argument('--update_model', action='store_true', default=False)
parser.add_argument('--contrastive', action='store_true', default=False)
parser.add_argument('--knn_eval_freq', default=1, type=int)
parser.add_argument('--distill_freq', default=5, type=int)
parser.add_argument('--saved_path', default='none', type=str)
parser.add_argument('--mode', default='normal', choices=['normal', 'frequency'])

## ssl setting
parser.add_argument('--temp', default=0.5, type=float)
parser.add_argument('--lr', default=0.06, type=float)
parser.add_argument('--wd', default=5e-4, type=float)
parser.add_argument('--cos', action='store_true', default=True)
parser.add_argument('--byol-m', default=0.996, type=float)

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
parser.add_argument('--ratio', default = 0.5, type=float, help='the ratio of D_public to the union of D_public and D_alternative'
                    , choices=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
###logging
parser.add_argument('--log_path', default='Experiments', type=str, help='path to save log')
parser.add_argument('--poison_knn_eval_freq', default=5, type=int)
parser.add_argument('--poison_knn_eval_freq_iter', default=1, type=int)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--trial', default='clean', type=str, choices=['clean', 'poisoned'])

###others
parser.add_argument('--distributed', action='store_true',
                    help='distributed training')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--seed', default=99, type=int,
                    help='seed for initializing training. ')



args = parser.parse_args()
wandb.init(sync_tensorboard=False,
        project="CTRL_SSL_Train",
        name='{}-{}-{}-{}-{}-{}'.format(args.dataset, args.part, args.method, args.arch, args.mode, args.unlabel) if args.part != '' 
        else '{}-{}-{}-{}-{}'.format(args.dataset, args.method, args.arch, args.mode, args.unlabel),
        config=args,
        # id='2ozylnnd'
        )


# for Logging
if args.debug: #### in the debug setting
        args.saved_path = os.path.join("./{}/test".format(args.log_path))
else:
    if  args.trial == 'clean':
        args.saved_path = os.path.join(
            "./{}/{}/{}-{}-{}-{}-{}-{}-{}".format(args.log_path, args.steal_dataset, args.dataset, args.part, args.method, args.arch, args.trial, args.unlabel, args.ratio)) if args.part != '' else os.path.join(
                "./{}/{}/{}-{}-{}-{}-{}-{}".format(args.log_path, args.steal_dataset, args.dataset, args.method, args.arch, args.trial, args.unlabel, args.ratio))
    else:
        args.saved_path = os.path.join("./{}/{}/{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(args.log_path, args.steal_dataset, args.dataset, args.method, args.arch, args.poison_ratio, args.magnitude, args.batch_size, args.lr, args.select, args.threat_model, args.trial, args.unlabel, args.ratio))


if not os.path.exists(args.saved_path):
    os.makedirs(args.saved_path)

# tb_logger = tb_logger.Logger(logdir=args.saved_path, flush_secs=2)

def main():
    print(args.saved_path)
    if args.seed is not None:
        # random.seed(args.seed)
        # torch.manual_seed(args.seed)
        set_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')


    main_worker(args.gpu,  args)

def main_worker(gpu,  args):

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating cnn model '{}'".format(args.arch))
    model = set_model(args)

    # constrcut trainer
    trainer = CLTrainer(args)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # create data loader
    train_loader, train_sampler, train_dataset, ft_loader, ft_sampler, test_loader, test_dataset, memory_loader, train_transform, ft_transform, test_transform = set_aug_diff(args)

    # create poisoning dataset
    if args.poisoning:
        poison_frequency_agent = PoisonFre(args, args.size, args.channel, args.window_size, args.trigger_position,  False,  True)
        poison = PoisonAgent(args, poison_frequency_agent, train_dataset, test_dataset, memory_loader, args.magnitude)

    # create optimizer
    optimizer = optim.SGD(model.parameters(),
                        lr=args.lr,
                        momentum=0.9,
                        weight_decay=args.wd)

    # resume
    if args.start_epoch > 0:
        checkpoint = torch.load(os.path.join(args.saved_path, f'epoch_{args.start_epoch}.pth.tar'), map_location='cuda:0')
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    # Train
    if args.mode == 'normal':
         trainer.train(model, optimizer, train_loader, test_loader, memory_loader, train_sampler, train_transform)


    elif args.mode == 'frequency':
         trainer.train_freq(model, optimizer, train_transform, poison)
        
    wandb.finish()


    raise NotImplementedError


if __name__ == '__main__':
    main()
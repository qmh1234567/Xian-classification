# -*- coding: utf-8 -*-
"""
基于 PyTorch resnet50 实现的图片分类代码
原代码地址：https://github.com/pytorch/examples/blob/master/imagenet/main.py
可以与原代码进行比较，查看需修改哪些代码才可以将其改造成可以在 ModelArts 上运行的代码
在ModelArts Notebook中的代码运行方法：
（0）准备数据
大赛发布的公开数据集是所有图片和标签txt都在一个目录中的格式
如果需要使用 torch.utils.data.DataLoader 来加载数据，则需要将数据的存储格式做如下改变：
1）划分训练集和验证集，分别存放为 train 和 val 目录；
2）train 和 val 目录下有按类别存放的子目录，子目录中都是同一个类的图片
prepare_data.py中的 split_train_val 函数就是实现如上功能，建议先在自己的机器上运行该函数，然后将处理好的数据上传到OBS
执行该函数的方法如下：
cd {prepare_data.py所在目录}
python prepare_data.py --input_dir '../datasets/train_data' --output_train_dir '../datasets/train_val/train' --output_val_dir '../datasets/train_val/val'

（1）从零训练
cd {main.py所在目录}
python main.py --data_url '../datasets/train_val' --train_url '../model_snapshots' --deploy_script_path './deploy_scripts' --   'resnet50' --num_classes 54 --workers 4 --epochs 6 --pretrained True --seed 0

（2）加载已有模型继续训练
cd {main.py所在目录}
python main.py --data_url '../datasets/train_val' --train_url '../model_snapshots' --deploy_script_path './deploy_scripts' --arch 'resnet50' --num_classes 54 --workers 4 --epochs 6 --seed 0 --resume '../model_snapshots/epoch_0_2.4.pth'

（3）评价单个pth文件
cd {main.py所在目录}
python main.py --data_url '../datasets/train_val' --train_url '../model_snapshots' --arch 'resnet50' --num_classes 54 --seed 0 --eval_pth '../model_snapshots/epoch_5_8.4.pth'
"""
import argparse
import os
import random
import shutil
import time
import warnings
from collections import OrderedDict

try:
    import moxing as mox
except:
    print('not use moxing')
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import utils

from torchsummary import summary
import transform
from prepare_data import prepare_data_on_modelarts
# 新增
import torch.optim.lr_scheduler as lr_scheduler
from efficientnet_pytorch import EfficientNet
from torch.utils.data.dataloader import default_collate  # 导入默认的拼接方式

import resnet_model_cbam
import numpy as np
from cutmix import CutMixCollator

from sklearn import svm


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


print(model_names)
print(torchvision.__version__)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', required=True,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-fc-times', '--lft', default=5, type=int,
                    metavar='LR', help='initial model last layer rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
parser.add_argument('--eval_pth', default='', type=str,
                    help='the *.pth model path need to be evaluated on validation set')
parser.add_argument('--pretrained', default=True, type=bool,
                    help='use pre-trained model or not')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# These arguments are added for adapting ModelArts
parser.add_argument('--num_classes', required=True, type=int, help='the num of classes which your task should classify')

parser.add_argument('--image-size', default=380, type=int, metavar='N',
                    help='the train image size')

parser.add_argument('--local_data_root', default='/cache/', type=str,
                    help='a directory used for transfer data between local path and OBS path')
parser.add_argument('--data_url', required=True, type=str, help='the training and validation data path')
parser.add_argument('--test_data_url', default='', type=str, help='the test data path')
parser.add_argument('--data_local', default='', type=str, help='the training and validation data path on local')
parser.add_argument('--test_data_local', default='', type=str, help='the test data path on local')
parser.add_argument('--train_url', required=True, type=str, help='the path to save training outputs')
parser.add_argument('--train_local', default='', type=str, help='the training output results on local')
parser.add_argument('--tmp', default='', type=str, help='a temporary path on local')
parser.add_argument('--deploy_script_path', default='', type=str,
                    help='a path which contain config.json and customize_service.py, '
                         'if it is set, these two scripts will be copied to {train_url}/model directory')
parser.add_argument('--save_vector',default=False,type=bool)



best_acc1 = 0


def main():
    args, unknown = parser.parse_known_args()
    args = prepare_data_on_modelarts(args)
    
    # exit()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
        # 保存训练集和验证集的特征及标签

def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)  # 用默认方式拼接过滤后的batch数据


def get_data_loader(args):
    # Data loading code
    transformations = transform.get_transforms(input_size=args.image_size,test_size=args.image_size)
    traindir = os.path.join(args.data_local, 'train')
    valdir = os.path.join(args.data_local, 'val')

    train_dataset = datasets.ImageFolder(
        traindir,
        transformations['val_train'],
    )
    val_dataset = datasets.ImageFolder(
        valdir, 
        transformations['val_test'],
    )
    
    # ImageFolder类会将traindir目录下的每个子目录名映射为一个label id，然后将该id作为模型训练时的标签
    # 比如，traindir目录下的子目录名分别是0~53，ImageFolder类将这些目录名当做class_name，再做一次class_to_idx的映射
    # 最终得到这样的class_to_idx：{"0": 0, "1":1, "10":2, "11":3, ..., "19": 11, "2": 12, ...}
    # 其中key是class_name，value是idx，idx就是模型训练时的标签
    # 因此我们在保存训练模型时，需要保存这种idx与class_name的映射关系，以便在做模型推理时，能根据推理结果idx得到正确的class_name
    idx_to_class = OrderedDict()
    for key, value in train_dataset.class_to_idx.items():
        idx_to_class[value] = key

    collater = CutMixCollator(1.0)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
         batch_size=args.batch_size, shuffle=True,
        #  collate_fn=collater,
        collate_fn=my_collate_fn,
        num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=True,
        collate_fn=my_collate_fn,
        num_workers=args.workers)
    
    return train_loader,val_loader,idx_to_class

# create model
def create_pretrained_model(args):
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))

        
        os.environ['TORCH_HOME'] = '../pre-trained_model/pytorch'  # 替换默认目录
        if not mox.file.exists('../pre-trained_model/pytorch/efficientnet-b7-dcc49843.pth'):
            mox.file.copy('s3://xianbucket/efficientnet-b7-dcc49843.pth',
                          '../pre-trained_model/pytorch/efficientnet-b7-dcc49843.pth')
            print('copy pre-trained model from OBS to: %s success' %
                  (os.path.abspath('../pre-trained_model/pytorch/efficientnet-b7-dcc49843.pth')))
        else:
            print('use exist pre-trained model at: %s' %
                  (os.path.abspath('../pre-trained_model/pytorch/efficientnet-b7-dcc49843.pth')))
        print("*"*10)
        # 加载模型结构
        # model = models.__dict__[args.arch](pretrained=False)
        model = EfficientNet.from_name('efficientnet-b7')
        # 加载模型参数
        model.load_state_dict(torch.load(os.path.abspath('../pre-trained_model/pytorch/efficientnet-b7-dcc49843.pth')))
        # 用Adam优化器
        model._fc.out_features = args.num_classes
        # print(model)


    else:
        print("=> creating model '{}'".format(args.arch))
        # model = models.__dict__[args.arch]()
        model = EfficientNet.from_name('efficientnet-b7')

    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, args.num_classes)
    return model

def create_model(args):
    # define model
    model = create_pretrained_model(args)

    # model = model.module
    print("load model's parameters success")

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            print("^"*100)
            model = torch.nn.DataParallel(model).cuda()
    return model


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # get data
    train_loader,val_loader,idx_to_class = get_data_loader(args)

    # create model
    model = create_model(args)

    # get optimizer
    optimizer = utils.get_optimizer(model,args)


    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = utils.LabelSmoothSoftmaxCE()
    
    # scheduler
    # Decay LR by a factor of 0.1 every 7 epochs
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.7,patience=5)
    # scheduler = utils.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
    #            step_size=step_size, mode='exp_range',
    #            gamma=0.99994)

    # optionally resume from a checkpoint
    if args.resume:
        # if os.path.isfile(args.resume):
        if mox.file.exists(args.resume) and (not mox.file.is_directory(args.resume)):
            if args.resume.startswith('s3://'):
                restore_model_name = args.resume.rsplit('/', 1)[1]
                mox.file.copy(args.resume, '/cache/tmp/' + restore_model_name)
                args.resume = '/cache/tmp/' + restore_model_name
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if args.resume.startswith('/cache/tmp/'):
                os.remove(args.resume)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    if args.distributed:
        train_sampler = None
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if args.eval_pth != '':
        if mox.file.exists(args.eval_pth) and (not mox.file.is_directory(args.eval_pth)):
            if args.eval_pth.startswith('s3://'):
                model_name = args.eval_pth.rsplit('/', 1)[1]
                mox.file.copy(args.eval_pth, '/cache/tmp/' + model_name)
                args.eval_pth = '/cache/tmp/' + model_name
            print("=> loading checkpoint '{}'".format(args.eval_pth))
            if args.gpu is None:
                checkpoint = torch.load(args.eval_pth)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.eval_pth, map_location=loc)
            if args.eval_pth.startswith('/cache/tmp/'):
                os.remove(args.eval_pth)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.eval_pth, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.eval_pth))

        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # print("optimizer:",optimizer.state_dict())
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # utils.adjust_learning_rate(optimizer, epoch, args)


        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        # if (epoch + 1) % args.print_freq == 0:
        if args.save_vector:
            acc1,test_loss,dict1 = validate(val_loader, model, criterion, epoch,args)
        else:
            acc1,test_loss = validate(val_loader, model, criterion, epoch,args)
            dict1 = None

        # remember best acc@1 and save checkpoint
        is_best = acc1.item() >= best_acc1

        # scheduler
        scheduler.step(test_loss)
        
        # save checkpoint
        best_acc1 = max(acc1.item(), best_acc1)
        pth_file_name = os.path.join(args.train_local, 'epoch_%s_%s.pth'
                                        % (str(epoch + 1), str(round(acc1.item(), 3))))
        
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'train_vec_dict': dict1,
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'idx_to_class': idx_to_class
            }, is_best, pth_file_name, args)

    # if args.epochs >= args.print_freq:
    utils.save_best_checkpoint(best_acc1, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    AverageMeter = utils.AverageMeter
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch+1))

    # switch to train mode
    model.train()

    end = time.time()  

    pth_file_name = os.path.join(args.train_local, 'train_epoch_%s.npy'
                                        % (str(epoch + 1)))

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu)
        target = target.cuda(args.gpu)

        # inputs, targets_a, targets_b, lam = transform.mixup_data(args,images, target, 1.)

        optimizer.zero_grad()
        # compute output
        output = model(images)

        loss = criterion(output, target) #之前用交叉熵损失函数
        # loss = transform.mixup_criterion(criterion,output,targets_a,targets_b,lam)
     
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # print(loss.item())
        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
   

def validate(val_loader, model, criterion,epoch, args):
    AverageMeter = utils.AverageMeter
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

     # 使用数组保存
    dict1 = {
            'vector':[],
            'label':[],
    }
    pth_file_name = os.path.join(args.train_local, 'epoch_%s.pt'
                                        % (str(epoch + 1)))


    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.save_vector and acc1 > 0.95:
                # 存入数组
                dict1['vector'].append(output)
                dict1['label'].append(target)

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    #  # 保存文件
    # if args.save_vector:
    #     torch.save(dict1, pth_file_name)
    #     # np.save(pth_file_name,dict1)
    #     if args.train_url.startswith('s3'):
    #             mox.file.copy(pth_file_name,
    #                         args.train_url + '/' + os.path.basename(pth_file_name))
    #             os.remove(pth_file_name)
    if args.save_vector:
        return top1.avg,losses.avg,dict1
    return top1.avg,losses.avg



if __name__ == '__main__':
    main()
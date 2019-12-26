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
python main.py --data_url '../datasets/train_val' --train_url '../model_snapshots' --deploy_script_path './deploy_scripts' --arch 'resnet50' --num_classes 54 --workers 4 --epochs 6 --pretrained True --seed 0

（2）加载已有模型继续训练
cd {main.py所在目录}
python main.py --data_url '../datasets/train_val' --train_url '../model_snapshots' --deploy_script_path './deploy_scripts' --arch 'resnet50' --num_classes 54 --workers 4 --epochs 6 --seed 0 --resume '../model_snapshots/epoch_0_2.4.pth'

（3）评价单个pth文件
cd {main.py所在目录}
python main.py --data_url '../datasets/train_val' --train_url '../model_snapshots' --arch 'resnet50' --num_classes 54 --seed 0 --eval_pth '../model_snapshots/epoch_5_8.4.pth'
"""
'''
1.train
python main.py --data_local './datasets/train_val' --local_data_root './cache' --arch 'resnext101_32x16d_wsl' --num_classes 54 --seed 0


'''
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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import utils
from transform import get_transforms
from torchsummary import summary
from prepare_data import prepare_data_on_modelarts
import resnet_model
# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

model_names = utils.get_modelnames()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=4, type=int,
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

parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

# 最新的checkpoint 加载路径
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--eval_pth', default='', type=str,
                    help='the *.pth model path need to be evaluated on validation set')

parser.add_argument('--pretrained', default=True, type=bool,
                    help='use pre-trained model or not')


parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')


parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# These arguments are added for adapting ModelArts
parser.add_argument('--num_classes', required=True, type=int, help='the num of classes which your task should classify')

parser.add_argument('--image-size', default=288, type=int, metavar='N',
                    help='the train image size')

parser.add_argument('--label_smoothing', default=0.1, type=int, metavar='N',
                    help='the train image size')


parser.add_argument('--local_data_root', required=True, type=str,
                    help='a directory used for transfer data between local path and OBS path')

parser.add_argument('--data_local', default='', type=str, help='the training and validation data path on local')
parser.add_argument('--test_data_local', default='', type=str, help='the test data path on local')
parser.add_argument('--train_local', default='', type=str, help='the training output results on local')

parser.add_argument('--tmp', default='', type=str, help='a temporary path on local')
parser.add_argument('--deploy_script_path', default='', type=str,
                    help='a path which contain config.json and customize_service.py, '
                         'if it is set, these two scripts will be copied to {train_url}/model directory')
# architecture
parser.add_argument('-a', '--arch', metavar='ARCH', required=True,
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')


best_acc1 = 0

def main():
    # 在单个gpu上训练
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args, unknown = parser.parse_known_args()
    args = prepare_data_on_modelarts(args)

    if args.seed is not None:
        random.seed(args.seed)
        # 保证每次启动进程的结果一样
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


    # Simply call main_worker function
    main_worker(args)

def get_data_loader(args):
    # Data loading code
    transformations = get_transforms(input_size=args.image_size,test_size=args.image_size)
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
         batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    
    return train_loader,val_loader,idx_to_class


    
def create_model(args):
    # define model

    # 加载模型结构
    resnet = utils.make_model(args)

    model = resnet_model.resnext101_32x8d()
    # 加载模型参数
    # 固定参数进行训练  一定要注意添加的位置
    model.fc = nn.Sequential(
         nn.Dropout(0.5),
         nn.Linear(2048,args.num_classes)
    )
    summary(model.cuda(),input_size=(3,288,288))
    exit()
    # 读取当前模型参数
    model_dict = model.state_dict()
    # 将预训练的网络中不属于模型的层的参数剃掉
    pretrained_dict = resnet.state_dict()  
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # model = model.module
    print("load model's parameters success")
    exit()
    return model


def main_worker(args):
    global best_acc1

    # 加载数据
    train_loader,val_loader,idx_to_class = get_data_loader(args)

    model = create_model(args)

    model = torch.nn.DataParallel(model).cuda()
    
    cudnn.benchmark = True

    
    # model.cuda()

    optimizer = utils.get_optimizer(model,args)

    # define loss function (criterion) and optimizer

    criterion = nn.CrossEntropyLoss().cuda()
    # criterion = utils.LabelSmoothSoftmaxCE()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.2, patience=5, verbose=False)

    # Resume
    start_epoch = args.start_epoch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc1 = checkpoint['best_acc1']
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    # else:
        # logger = Logger(os.path.join())

    for epoch in range(start_epoch, args.epochs):
        # utils.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss,train_acc,train_5 = train(train_loader, model, criterion, optimizer, epoch, args)
        # evaluate on validation set
        test_loss,test_acc,test_5= validate(val_loader, model, criterion, args)
        
        scheduler.step(test_loss)

        print('train_loss:%f, val_loss:%f, train_acc:%f, train_5:%f, val_acc:%f, val_5:%f' % (train_loss, test_loss, train_acc, train_5, test_acc, test_5))


        # remember best acc@1 and save checkpoint
        is_best = test_acc.item() >= best_acc1
        best_acc1 = max(test_acc.item(), best_acc1)
        pth_file_name = os.path.join(args.train_local, 'epoch_%s_%s.pth'
                                        % (str(epoch + 1), str(round(test_acc.item(), 3))))

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'idx_to_class': idx_to_class
        }, is_best, pth_file_name, args)

 
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
    for i, (images, target) in enumerate(train_loader):

        # print(images.shape)
        # print(target.size())
        # exit()

        # measure data loading time
        data_time.update(time.time() - end)

        
        images = images.cuda()
        target = target.cuda(async=True)
       
        images,target = torch.autograd.Variable(images),torch.autograd.Variable(target)

        # forward
        output = model(images)

        loss = criterion(output, target)
        # print(loss.item())
        # exit()

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output.data, target.data, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return (losses.avg,top1.avg,top5.avg)


def validate(val_loader, model, criterion, args):
    global best_acc1
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

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            images,target = torch.autograd.Variable(images),torch.autograd.Variable(target)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output.data, target.data, topk=(1, 5))
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

    return losses.avg,top1.avg,top5.avg


if __name__ == '__main__':
    main()
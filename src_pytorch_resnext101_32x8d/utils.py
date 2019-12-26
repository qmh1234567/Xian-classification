import torch
import os
import torch.nn as nn
import torchvision.models as models
try:
    import moxing as mox
except:
    print('not use moxing')



def get_optimizer(model,args):
    # params_to_update = model.parameters()
    count = 0
    para_optim = []
    # 冻结部分参数
    for name,param in model.named_parameters():
        # print("name=",name)
        count += 1
        if count > 200 or 'ca' in name or 'sa' in name:  # 200
            if 'ca' in name or 'sa' in name:
                print(name)
            para_optim.append(param)
        else:
            param.requires_grad = False

    print("count=",count)
    # parameters = []
    # for name,param in model.named_parameters():
    #     if 'fc' in name or 'class' in name or 'last_linear' in name or 'ca' in name or 'sa' in name:
    #         parameters.append({'params':param,'lr':args.lr*args.lr_fc_times})
    #     else:
    #         parameters.append({'params':param,'lr':args.lr})
    # 不同学习率设置
    # pretrained_params = list(map(id,model.add_layers.parameters()))
    # base_params = filter(lambda p: id(p) not in pretrained_params,model.parameters())
    optimizer = torch.optim.SGD(
                                # params_to_update,
                                filter(lambda p: p.requires_grad,model.parameters()),
                                # model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(
    #                             params_to_update,
    #                             # model.parameters(),
    #                             args.lr,
    #                             betas=(0.9, 0.999), 
    #                             eps=1e-9)
    # optimizer = torch.optim.Adam(parameters,lr=args.lr,betas=(0.9, 0.999), eps=1e-9)

    return optimizer


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})' # '{name} {val:.f} ({avg:.f})
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename, args):
    if  is_best:
        # print("&"*100)
        torch.save(state, filename)
        if args.train_url.startswith('s3'):
            mox.file.copy(filename,
                          args.train_url + '/' + os.path.basename(filename))
            os.remove(filename)


def save_best_checkpoint(best_acc1, args):
    best_acc1_suffix = '%s.pth' % str(round(best_acc1, 3))
    pth_files = mox.file.list_directory(args.train_url)
    # 选出得分最高的模型
    for pth_name in pth_files:
        if pth_name.endswith(best_acc1_suffix):
            break
    

    # mox.file可兼容处理本地路径和OBS路径
    if not mox.file.exists(os.path.join(args.train_url, 'model')):
        mox.file.mk_dir(os.path.join(args.train_url, 'model'))

    mox.file.copy(os.path.join(args.train_url, pth_name), os.path.join(args.train_url, 'model/model_best.pth'))


    mox.file.copy(os.path.join(args.deploy_script_path, 'config.json'),
                  os.path.join(args.train_url, 'model/config.json'))

    mox.file.copy(os.path.join(args.deploy_script_path, 'customize_service.py'),
                  os.path.join(args.train_url, 'model/customize_service.py'))

    mox.file.copy(os.path.join(args.deploy_script_path, 'Knn.py'),
                  os.path.join(args.train_url, 'model/Knn.py'))

    mox.file.copy(os.path.join(args.deploy_script_path, 'resnet_model_cbam.py'),
                  os.path.join(args.train_url, 'model/resnet_model_cbam.py'))

                  
    if mox.file.exists(os.path.join(args.train_url, 'model/config.json')) and \
            mox.file.exists(os.path.join(args.train_url, 'model/customize_service.py')):
        print('copy config.json and customize_service.py success')
    else:
        print('copy config.json and customize_service.py failed')


# # 标签平滑 + crossentropy_loss
class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
    
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)

        if self.reduction=='mean':
            loss1=-torch.sum(torch.sum(logs*lb_one_hot, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss1 = -torch.sum(logs*lb_one_hot, dim=1)

        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0

        if self.reduction == 'mean':
            loss = -torch.sum(torch.sum(logs*label, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss = -torch.sum(logs*label, dim=1)
        return (loss+loss1)/2
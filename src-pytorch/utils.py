import torch
import os
import torch.nn as nn
import torchvision
import torchvision.models as models
# 引入自定义的模型
import models as customized_models
from torchsummary import summary
import torch.nn.functional as F


def get_modelnames():
    default_model_names = sorted(name for name in models.__dict__ 
                                if name.islower() and not name.startswith("__")
                                and callable(models.__dict__[name]))
    customized_models_names = sorted(name for name in customized_models.__dict__
                                if not name.startswith("__") 
                                and callable(customized_models.__dict__[name]))
    for name in customized_models.__dict__:
        if not name.startswith("__") and callable(customized_models.__dict__[name]):
            models.__dict__[name] = customized_models.__dict__[name]
    print(default_model_names)
    model_names = default_model_names + customized_models_names
    print(model_names)
    return model_names



# 构建模型
def make_model(args):
    _ = get_modelnames()
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        # 加载预先训练的模型  pretrained=True表示读取网络结构和预训练模型，False表示只加载网络结构
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Dropout(0.2),
    #     nn.Linear(num_ftrs, args.num_classes)
    # )
    return model




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
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0

        if self.reduction == 'mean':
            loss = -torch.sum(torch.sum(logs*label, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss = -torch.sum(logs*label, dim=1)
        return loss



def get_optimizer(model,args):
    parameters = []
    
    for name,param in model.named_parameters():
        if 'fc' in name or 'class' in name or 'last_linear' in name or 'ca' in name or 'sa' in name:
            parameters.append({'params':param,'lr':args.lr*args.lr_fc_times})
        else:
            parameters.append({'params':param,'lr':args.lr})

    optimizer = torch.optim.SGD( 
                                parameters,
                                # model.parameters(),
                                args.lr,nesterov=True,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
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


# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


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
    cur_path = os.path.join(args.train_local,'_'+'model_cur.pth')
    torch.save(state['state_dict'],cur_path)
    # 保存最优模型
    if not is_best:
        torch.save(state, filename)



import os
from matplotlib import pyplot as plt

import torch
from torch.nn import init


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


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def init_weights_normal(m):
    """Normal Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)


def init_weights_xavier(m):
    """Xavier Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)


def init_weights_kaiming(m):
    """Kaiming He Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')


def get_lr_scheduler(lr_scheduler, optimizer):
    """Learning Rate Scheduler"""
    if lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    else:
        raise NotImplementedError

    return scheduler


def plot_metrics(losses, accs, path, model, dataset, num_classes):
    """Plot and Save Figure on Loss and Accuracy"""
    (train_losses, val_losses) = losses
    (train_top1_acc, train_top5_acc, val_top1_acc, val_top5_acc) = accs

    # Plot on Loss #
    plt.figure(1)
    plt.plot(train_losses, label='Train Loss', alpha=0.5)
    plt.plot(val_losses, label='Val Loss', alpha=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend(loc='best')
    plt.title('{} Loss on {} {} Dataset'.format(model.__class__.__name__, dataset.upper(), num_classes))
    plt.savefig(os.path.join(path, '{} Loss on {} {} Dataset.png'.format(model.__class__.__name__, dataset.upper(), num_classes)))

    # Plot on Accuracy #
    plt.figure(2)
    plt.plot(train_top1_acc, label='Train Top1 Accuracy', alpha=0.5)
    plt.plot(train_top5_acc, label='Train Top5 Accuracy', alpha=0.5)
    plt.plot(val_top1_acc, label='Val Top1 Accuracy', alpha=0.5)
    plt.plot(val_top5_acc, label='Val Top5 Accuracy', alpha=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend(loc='best')
    plt.title('{} Accuray on {} {} Dataset'.format(model.__class__.__name__, dataset.upper(), num_classes))
    plt.savefig(os.path.join(path, '{} Accuracy on {} {} Dataset.png'.format(model.__class__.__name__, dataset.upper(), num_classes)))
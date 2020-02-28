import torch

class AverageMeter(object):
    #数値を保持し、平均を返すクラス

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):   #数値を初期化
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):  #平均値を更新
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):        #strとして呼び出されると、名前・数値・平均値を表示する
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    #Averagemeterクラスの入力を受け、表示を行うクラス

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]

        # show current values and average values
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        # format the number of digits for string
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def get_mean(norm_value=255):
    # mean of imagenet
    return [
        123.675 / norm_value, 116.28 / norm_value,
        103.53 / norm_value
    ]


def get_std(norm_value=255):
    # std fo imagenet
    return [
        58.395 / norm_value, 57.12 / norm_value,
        57.375 / norm_value
    ]

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

def reqg_config(model, modelname, CONFIG, reqg_lastonly=False):
    if reqg_lastonly:
        if modelname == "googlenet":
            for p in model.head.parameters():
                p.requires_grad = False
            for p in model.i3.parameters():
                p.requires_grad = False
            for p in model.i4.parameters():
                p.requires_grad = False
            for p in model.i5.parameters():
                p.requires_grad = False
            for p in model.fc1.parameters():
                p.requires_grad = True
            for p in model.fc2.parameters():
                p.requires_grad = True
            
            
        if modelname == "resnet":
            for p in model.resnet.conv1.parameters():
                p.requires_grad = False
            for p in model.resnet.layer1.parameters():
                p.requires_grad = False
            for p in model.resnet.layer2.parameters():
                p.requires_grad = False
            for p in model.resnet.layer3.parameters():
                p.requires_grad = False
            for p in model.resnet.layer4.parameters():
                p.requires_grad = False
            for p in model.fc1.parameters():
                p.requires_grad = True
            for p in model.fc2.parameters():
                p.requires_grad = True
    
    else:
        if modelname == "googlenet":
            for p in model.head.parameters():
                p.requires_grad = CONFIG.reqg_head
            for p in model.i3.parameters():
                p.requires_grad = CONFIG.reqg_head
            for p in model.i4.parameters():
                p.requires_grad = CONFIG.reqg_i4
            for p in model.i5.parameters():
                p.requires_grad = CONFIG.reqg_i5
            for p in model.fc1.parameters():
                p.requires_grad = CONFIG.reqg_last
            for p in model.fc2.parameters():
                p.requires_grad = CONFIG.reqg_last
            
            
        if modelname == "resnet":
            for p in model.resnet.conv1.parameters():
                p.requires_grad = CONFIG.reqg_head
            for p in model.resnet.layer1.parameters():
                p.requires_grad = CONFIG.reqg_layer1
            for p in model.resnet.layer2.parameters():
                p.requires_grad = CONFIG.reqg_layer2
            for p in model.resnet.layer3.parameters():
                p.requires_grad = CONFIG.reqg_layer3
            for p in model.resnet.layer4.parameters():
                p.requires_grad = CONFIG.reqg_layer4
            for p in model.fc1.parameters():
                p.requires_grad = CONFIG.reqg_last
            for p in model.fc2.parameters():
                p.requires_grad = CONFIG.reqg_last
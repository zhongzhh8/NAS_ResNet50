import os
import torch
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import utils_jump
import dataset
import torch.optim as optim
import time
from PIL import Image
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser(description='BlockDrop Training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')

parser.add_argument('--deviceid', type=list, default=[5], help='')

parser.add_argument('--cv_dir', default='checkpoints_early/teachers/cifar10/r50_48B/12_05/default1', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--binary_bits', type=int, default=48, help='length of hashing binary')
parser.add_argument('--margin', type=float, default=20, help='margin of triplet loss')


parser.add_argument('--lr_step', type=int, default=50, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=150, help='total epochs to run')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='lr *= lr_decay_ratio after epoch_steps')

parser.add_argument('--samples_each_label', type=int, default=10, help='number of samples each label in a batch')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.makedirs(args.cv_dir)
utils_jump.save_args(__file__, args)

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(torchdata.Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            if line == '':
                break
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(' ')
            imgs.append(words)
        self.imgs = imgs


    def __getitem__(self, index):
        words = self.imgs[index]
        img = self.loader(words[0])
        if self.transform is not None:
            img = self.transform(img)
        label = int(words[1])
        return img,label

    def __len__(self):
        return len(self.imgs)

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        triplets = triplets.to(device)

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


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

def compute_result(dataloader):
    bs, clses = [], []
    total_time = 0
    batch_time = AverageMeter()
    for img, cls in tqdm.tqdm(dataloader):
        clses.append(cls)
        inputs, targets = Variable(img).to(device), Variable(cls).to(device)

        time_start = time.time()
        with torch.no_grad():
            _,_,_,out = rnet(inputs)
            preds = BranchLast(out)
        time_used = time.time() - time_start
        total_time += time_used
        batch_time.update(time_used)
        bs.append(preds.data.cpu())
    return torch.sign(torch.cat(bs)), torch.cat(clses), total_time, batch_time.avg


def compute_result_all(dataloader):
    bs1, bs2, bs3, bs4, clses = [], [], [], [], []

    total_time = 0
    batch_time = AverageMeter()
    for img, cls in tqdm.tqdm(dataloader):
        clses.append(cls)
        inputs, targets = Variable(img).to(device), Variable(cls).to(device)

        time_start = time.time()
        with torch.no_grad():
            _,_,_,out = rnet(inputs)
            code = BranchLast(out)

        time_used = time.time() - time_start
        total_time += time_used
        batch_time.update(time_used)
        bs1.append(code.data.cpu())
        bs2.append(code.data.cpu())
        bs3.append(code.data.cpu())
        bs4.append(code.data.cpu())
    return torch.sign(torch.cat(bs1)), torch.sign(torch.cat(bs2)), torch.sign(torch.cat(bs3)), torch.sign(torch.cat(bs4)), torch.cat(clses), total_time, batch_time.avg


def test(epoch):
    f_test = open(args.cv_dir + '/a_test_record.txt', 'a')
    for width_mult in sorted(width_mult_list, reverse=True):  # 1x,0.75x,0.5x,0.25x依次训练，用1x的soft target蒸馏训练其他
        if width_mult != 1.0:  # 只训练1.0x
            continue
        def fn(m):
            setattr(m, 'width_mult', width_mult)  # width_mult在SlimmableConv2d和SwitchableBatchNorm2d里有
        rnet.apply(fn)
        BranchLast.apply(fn)
        rnet.eval()
        BranchLast.eval()

        tst_code1, tst_code2, tst_code3, tst_code4, tst_label, tst_time, tst_avg_time = compute_result_all(test_loader)
        db_binary, db_label, db_time, db_avg_time = compute_result(database_loader)
        mAP1=mAP2=mAP3=-1
        mAP4 = utils_jump.compute_mAP(db_binary, tst_code4, db_label, tst_label)

        if width_mult==1.0:
            torch.save({
                'rnet_state_dict': rnet.state_dict(),
                'branch_last_state_dict': BranchLast.state_dict(),
            }, args.cv_dir+'/E_%03d_mAP_%.5f_%.5f_%.5f_%.5f.pth'%(epoch,mAP1,mAP2,mAP3,mAP4))

        print("width %.2f epoch: %d, retrieval mAP1: %.6f, retrieval mAP2: %.6f, retrieval mAP3: %.6f, retrieval mAP4: %.6f, " % (width_mult,epoch, mAP1, mAP2, mAP3, mAP4))
        f_test.write('width %.2f E_%d_mAP_%.5f_%.5f_%.5f_%.5f.t7' % (width_mult,epoch, mAP1, mAP2, mAP3, mAP4) + '\n')
    f_test.close()

def Load_data():
    root_dir = './data/cifar10/'
    train_dir = root_dir + 'train_500.txt'
    database_dir = root_dir + 'database_5900.txt'
    test_dir = root_dir + 'test_100.txt'
    transform_train = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    transform_test = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor()])

    trainset = MyDataset(txt=train_dir, transform=transform_train)
    database = MyDataset(txt=database_dir, transform=transform_test)
    testset = MyDataset(txt=test_dir, transform=transform_test)

    train_loader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    database_loader = torchdata.DataLoader(database, batch_size=args.batch_size, shuffle=False)
    test_loader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    train_batch_sampler = dataset.BalancedBatchSampler_cifa10_500(trainset, n_classes=10,
                                                                  n_samples=args.samples_each_label)
    online_train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=train_batch_sampler, num_workers=4)
    return train_loader,test_loader,database_loader,online_train_loader

def Load_checkpoint():
    # 加载s-resnet50的预训练模型
    rnet_checkpoint = torch.load('s_resnet50_0.25_0.5_0.75_1.0.pt')['model']
    rnet.load_state_dict(rnet_checkpoint)
    # #加载resnet50的预训练模型
    # from torchvision.models import resnet50
    # rnet_dict=rnet.state_dict()
    # rnet50=resnet50(pretrained=True)
    # rnet50_dict=rnet50.state_dict()
    # new_rnet_dict={}
    #
    # for (k,v) in rnet_dict.items(): #先去掉多余的BN层参数
    #     if 'bn.0' in k or 'bn.1' in k or 'bn.2' in k:
    #         continue
    #     # print(k,v.size())
    #     new_rnet_dict[k]=v
    #
    # for ((k,v),(k50,v50)) in zip(new_rnet_dict.items(),rnet50_dict.items()):#变成rnet的key，rnet50的value
    #     new_rnet_dict[k] = v50
    # rnet_dict.update(new_rnet_dict) #更新rnet的部分参数（不重合的那些BN层不更新）
    # rnet.load_state_dict(rnet_dict)


#--------------------------------------------------------------------------------------------------------#
device = torch.device("cuda:" + str(args.deviceid[0]) if torch.cuda.is_available() else "cpu")  # device configuration
train_loader,test_loader,database_loader,online_train_loader=Load_data()

online_triplet_loss = OnlineTripletLoss(args.margin,  dataset.AllTripletSelector())

from models import s_resnet
rnet=s_resnet.Model()
BranchLast=s_resnet.BranchLast(bits=args.binary_bits)

rnet.to(device)
rnet = torch.nn.DataParallel(rnet, device_ids=args.deviceid)  # for multi gpu
BranchLast.to(device)
BranchLast=torch.nn.DataParallel(BranchLast, device_ids=args.deviceid)

Load_checkpoint()

width_mult_list=[0.25, 0.5, 0.75, 1.0]
start_epoch = 0
optimizer = optim.Adam(list(rnet.parameters())+list(BranchLast.parameters()), lr=args.lr, weight_decay=args.wd)
lr_scheduler = utils_jump.LrScheduler(optimizer, args.lr, args.lr_decay_ratio, args.lr_step)
total_tst_time =0
test_cnt = 0

for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    lr_scheduler.adjust_learning_rate(epoch)
    for width_mult in sorted(width_mult_list, reverse=True):  # 1x,0.75x,0.5x,0.25x依次训练，用1x的soft target蒸馏训练其他
        if width_mult!=1.0: #只训练1.0x
            continue
        def fn(m):
            setattr(m, 'width_mult', width_mult) #width_mult在SlimmableConv2d和SwitchableBatchNorm2d里有
        rnet.apply(fn)
        BranchLast.apply(fn)

        rnet.train()
        BranchLast.train()

        accum_loss = 0
        batch_time = AverageMeter()
        for batch_idx, (inputs, targets) in enumerate(online_train_loader):
            inputs, targets = Variable(inputs).to(device), Variable(targets).to(device)

            time_t = time.time()
            _,_,_,out=rnet(inputs)
            code=BranchLast(out)

            batch_time.update(time.time() - time_t)

            loss, len_triplets = online_triplet_loss.forward(code, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accum_loss += loss

        print("%.2fx epoch: %d, accum_loss: %.6f, average batch time: %.6f" % (width_mult,epoch, accum_loss, batch_time.avg))

    if epoch % 10 == 0 and epoch > 0:
        test(epoch)


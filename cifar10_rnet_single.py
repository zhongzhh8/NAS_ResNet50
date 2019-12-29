import os
# from tensorboard_logger import configure, log_value
import torch
# from tensorboardX import SummaryWriter
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import utils_jump
import dataset
import torch.optim as optim
from itertools import combinations
import time
from PIL import Image
import torchvision.transforms as transforms

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


import argparse
parser = argparse.ArgumentParser(description='BlockDrop Training')

parser.add_argument('--cudaid', type=str, default='0', help='')
parser.add_argument('--deviceid', type=list, default=[0, 1,2,3], help='')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
parser.add_argument('--model', default='R34_SUN', help='R<depth>_<dataset> see utils_jump.py for a list of configurations')
parser.add_argument('--data_dir', default='../../../disk2/zenghaien/imagenet/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load rnet+agent from')
parser.add_argument('--pretrained', default=None, help='pretrained policy model checkpoint (from curriculum training)')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epoch_step', type=int, default=200, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=2000, help='total epochs to run')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='lr *= lr_decay_ratio after epoch_steps')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
# parser.add_argument('--joint', action ='store_true', default=True, help='train both the policy network and the resnet')
parser.add_argument('--penalty', type=float, default=-5, help='gamma: reward for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--alpha2', type=float, default=0.01, help='weighting of regularizer')
parser.add_argument('--binary_bits', type=int, default=12, help='length of hashing binary')

parser.add_argument('--margin', type=float, default=1, help='margin of triplet loss')#实际上在代码那里写的

parser.add_argument('--samples_each_label', type=int, default=20, help='number of samples each label in a batch')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
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

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

class OnlineTripletLoss_jump(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss_jump, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, code1, code2, code3, code4, target):

        # triplets1 = self.triplet_selector.get_triplets(code1, target)
        # triplets2 = self.triplet_selector.get_triplets(code2, target)
        # triplets3 = self.triplet_selector.get_triplets(code3, target)
        triplets4 = self.triplet_selector.get_triplets(code4, target)

        # if code1.is_cuda:
        #     triplets1 = triplets1.cuda()
        # if code2.is_cuda:
        #     triplets2 = triplets2.cuda()
        # if code3.is_cuda:
        #     triplets3 = triplets3.cuda()
        if code4.is_cuda:
            triplets4 = triplets4.cuda()

        losses = 0.0
        ap_distances1 = (code1[triplets4[:, 0]] - code1[triplets4[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances1 = (code1[triplets4[:, 0]] - code1[triplets4[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses += F.relu(ap_distances1 - an_distances1 + 10).pow(2).mean()

        ap_distances2 = (code2[triplets4[:, 0]] - code2[triplets4[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances2 = (code2[triplets4[:, 0]] - code2[triplets4[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses += F.relu(ap_distances2 - an_distances2 + 10).pow(2).mean()

        ap_distances3 = (code3[triplets4[:, 0]] - code3[triplets4[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances3 = (code3[triplets4[:, 0]] - code3[triplets4[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses += F.relu(ap_distances3 - an_distances3 + 10).pow(2).mean()

        ap_distances4 = (code4[triplets4[:, 0]] - code4[triplets4[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances4 = (code4[triplets4[:, 0]] - code4[triplets4[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses += F.relu(ap_distances4 - an_distances4 + 10).pow(2).mean()


        return losses, len(triplets4)

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


def train_online_triplet(epoch):
    rnet.train()
    BranchFirst.train()
    BranchSecond.train()
    BranchThird.train()
    BranchLast.train()

    accum_loss = 0
    batch_time = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(online_train_loader):

        inputs, targets = Variable(inputs), Variable(targets).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()

        time_t = time.time()
        # preds_sample = rnet.forward_last(inputs)
        out1,out2,out3,out4 = rnet.forward_all(inputs)
        code1 = BranchFirst.forward(out1)
        code1 = torch.tanh(code1)
        code2 = BranchSecond.forward(out2)
        code2 = torch.tanh(code2)
        code3 = BranchThird.forward(out3)
        code3 = torch.tanh(code3)
        code4 = BranchLast.forward(out4)
        code4 = torch.tanh(code4)
        batch_time.update(time.time() - time_t)

        loss, len_triplets = online_triplet_loss.forward(code1, code2, code3, code4, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accum_loss += loss

    # logger.add_scalar('accum_loss', accum_loss, epoch)
    print("epoch: %d, accum_loss: %.6f, average batch time: %.6f" % (epoch, accum_loss, batch_time.avg))

def compute_result(dataloader):
    bs, clses = [], []
    total_time = 0
    batch_time = AverageMeter()
    for img, cls in dataloader:
        clses.append(cls)
        inputs, targets = Variable(img, volatile=True), Variable(cls).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()

        time_start = time.time()
        preds = rnet.forward_last(inputs)
        preds = BranchLast.forward(preds)
        # preds = F.tanh(preds)
        time_used = time.time() - time_start
        total_time += time_used
        batch_time.update(time_used)
        bs.append(preds.data.cpu())
    return torch.sign(torch.cat(bs)), torch.cat(clses), total_time, batch_time.avg


def compute_result_all(dataloader):
    bs1, bs2, bs3, bs4, clses = [], [], [], [], []

    total_time = 0
    batch_time = AverageMeter()
    for img, cls in dataloader:
        clses.append(cls)
        inputs, targets = Variable(img), Variable(cls).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()

        time_start = time.time()
        with torch.no_grad():
            code1,code2,code3,code4 = rnet.forward_all(inputs)
            code1 = BranchFirst(code1)
            code2 = BranchSecond(code2)
            code3 = BranchThird(code3)
            code4 = BranchLast(code4)
        time_used = time.time() - time_start
        total_time += time_used
        batch_time.update(time_used)
        bs1.append(code1.data.cpu())
        bs2.append(code2.data.cpu())
        bs3.append(code3.data.cpu())
        bs4.append(code4.data.cpu())
    return torch.sign(torch.cat(bs1)), torch.sign(torch.cat(bs2)), torch.sign(torch.cat(bs3)), torch.sign(torch.cat(bs4)), torch.cat(clses), total_time, batch_time.avg


def test_all(epoch):

    rnet.eval()
    BranchFirst.eval()
    BranchSecond.eval()
    BranchThird.eval()
    BranchLast.eval()

    t1=time.time()
    tst_code1, tst_code2, tst_code3, tst_code4, tst_label, tst_time, tst_avg_time = compute_result_all(test_loader)
    t2 = time.time()
    db_code1, db_code2, db_code3, db_code4, db_label, db_time, db_avg_time = compute_result_all(database_loader)
    t3 = time.time()
    print('t2-t1=',t2-t1)
    print('t3-t2=', t3 - t2)

    # db_binary, db_label, db_time, db_avg_time = compute_result(database_loader)
    mAP1 = utils_jump.compute_mAP(db_code1, tst_code1, db_label, tst_label)
    mAP2 = utils_jump.compute_mAP(db_code2, tst_code2, db_label, tst_label)
    mAP3 = utils_jump.compute_mAP(db_code3, tst_code3, db_label, tst_label)
    mAP4 = utils_jump.compute_mAP(db_code4, tst_code4, db_label, tst_label)

    print("epoch: %d, retrieval mAP1: %.6f, retrieval mAP2: %.6f, retrieval mAP3: %.6f, retrieval mAP4: %.6f, " % (epoch, mAP1, mAP2, mAP3, mAP4))

    # save the model
    rnet_state_dict = rnet.module.state_dict() if args.parallel else rnet.state_dict()
    branch_first_state = BranchFirst.module.state_dict() if args.parallel else BranchFirst.state_dict()
    branch_second_state = BranchSecond.module.state_dict() if args.parallel else BranchSecond.state_dict()
    branch_third_state = BranchThird.module.state_dict() if args.parallel else BranchThird.state_dict()
    branch_last_state = BranchLast.module.state_dict() if args.parallel else BranchLast.state_dict()

    torch.save(rnet_state_dict, args.cv_dir+'/ckpt_E_%d_mAP_%.5f_%.5f_%.5f_%.5f.t7'%(epoch,mAP1,mAP2,mAP3,mAP4))
    torch.save(branch_first_state, args.cv_dir+'/ckpt_E_%d_mAP_%.5f_%.5f_%.5f_%.5f_branchFirst.t7'%(epoch,mAP1,mAP2,mAP3,mAP4))
    torch.save(branch_second_state, args.cv_dir+'/ckpt_E_%d_mAP_%.5f_%.5f_%.5f_%.5f_branchSecond.t7'%(epoch,mAP1,mAP2,mAP3,mAP4))
    torch.save(branch_third_state, args.cv_dir+'/ckpt_E_%d_mAP_%.5f_%.5f_%.5f_%.5f_branchThird.t7'%(epoch,mAP1,mAP2,mAP3,mAP4))
    torch.save(branch_last_state, args.cv_dir+'/ckpt_E_%d_mAP_%.5f_%.5f_%.5f_%.5f_branchLast.t7'%(epoch,mAP1,mAP2,mAP3,mAP4))

    return tst_time,db_time

#--------------------------------------------------------------------------------------------------------#
root_dir = './data/cifar10/'
train_dir = root_dir+'train_500.txt'
database_dir = root_dir+'database_5900.txt'
test_dir = root_dir+'test_100.txt'
# transform_train_32 = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
transform_train = transforms.Compose([transforms.Resize((256, 256)),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
# transform_test_32 = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.Resize((256, 256)),transforms.CenterCrop(224),transforms.ToTensor()])

trainset = MyDataset(txt=train_dir, transform=transform_train)
database = MyDataset(txt=database_dir, transform=transform_test)
testset = MyDataset(txt=test_dir, transform=transform_test)

train_loader = torchdata.DataLoader(trainset, batch_size=args.batch_size,shuffle=True)
database_loader = torchdata.DataLoader(database, batch_size=args.batch_size,shuffle=False)
test_loader = torchdata.DataLoader(testset, batch_size=args.batch_size,shuffle=False)

train_batch_sampler = dataset.BalancedBatchSampler_cifa10_500(trainset, n_classes=10, n_samples=args.samples_each_label)
# online_train_loader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
online_train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=train_batch_sampler, num_workers=4)
# online_test_loader = torch.utils.data.DataLoader(testset, batch_sampler=test_batch_sampler)

# online_triplet_loss = OnlineTripletLoss(args.margin, dataset.RandomNegativeTripletSelector(args.margin))
# online_triplet_loss = OnlineTripletLoss_jump(args.margin, dataset.SemihardNegativeTripletSelector(args.margin))
# online_triplet_loss = OnlineTripletLoss_jump(args.margin, dataset.RandomNegativeTripletSelector(args.margin))
# online_triplet_loss = OnlineTripletLoss(args.margin, dataset.HardestNegativeTripletSelector(args.margin))
online_triplet_loss = OnlineTripletLoss_jump(args.margin, dataset.AllTripletSelector())
# logger = SummaryWriter()

from models import resnet_standard
# rnet = resnet_standard.resnet34_c10(pretrained=False)
# rnet = resnet_standard.resnet10(pretrained=True)
rnet = resnet_standard.resnet18(pretrained=True)
BranchFirst = resnet_standard.BranchFirst(inplanes=64, expansion=1, bits=12)
BranchSecond = resnet_standard.BranchSecond(inplanes=128, expansion=1, bits=12)
BranchThird = resnet_standard.BranchThird(inplanes=256, expansion=1, bits=12)
BranchLast = resnet_standard.BranchLast(inplanes=512, expansion=1, bits=12)

# rnet_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/cifar10/R10_48B/ckpt_E_180_mAP_0.16886_0.18033_0.15826_0.78290.t7'
# rnet_checkpoint = torch.load(rnet_checkpoint)
# rnet.load_state_dict(rnet_checkpoint)
#
# b1_checkpoint = 'cv/pretrained/R34_C10/ckpt_E_80_mAP_0.13162_0.08831_0.13682_0.66700_branchFirst.t7'
# b1_checkpoint = torch.load(b1_checkpoint)
# BranchFirst.load_state_dict(b1_checkpoint)
#
# b2_checkpoint = 'cv/pretrained/R34_C10/ckpt_E_80_mAP_0.13162_0.08831_0.13682_0.66700_branchSecond.t7'
# b2_checkpoint = torch.load(b2_checkpoint)
# BranchSecond.load_state_dict(b2_checkpoint)
#
# b3_checkpoint = 'cv/pretrained/R34_C10/ckpt_E_80_mAP_0.13162_0.08831_0.13682_0.66700_branchThird.t7'
# b3_checkpoint = torch.load(b3_checkpoint)
# BranchThird.load_state_dict(b3_checkpoint)
#
# b4_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/cifar10/R10_48B/ckpt_E_180_mAP_0.16886_0.18033_0.15826_0.78290_branchLast.t7'
# b4_checkpoint = torch.load(b4_checkpoint)
# BranchLast.load_state_dict(b4_checkpoint)

start_epoch = 0

rnet.cuda()
BranchFirst.cuda()
BranchSecond.cuda()
BranchThird.cuda()
BranchLast.cuda()


device = torch.device("cuda:" + args.cudaid if torch.cuda.is_available() else "cpu")  # device configuration

# rnet.to(device)
# rnet = torch.nn.DataParallel(rnet, device_ids=args.deviceid)  # for multi gpu

# BranchFirst.to(device)
# BranchFirst = torch.nn.DataParallel(BranchFirst, device_ids=args.deviceid)  # for multi gpu
# BranchSecond.to(device)
# BranchSecond = torch.nn.DataParallel(BranchSecond, device_ids=args.deviceid)  # for multi gpu
# BranchThird.to(device)
# BranchThird = torch.nn.DataParallel(BranchThird, device_ids=args.deviceid)  # for multi gpu
# BranchLast.to(device)
# BranchLast = torch.nn.DataParallel(BranchLast, device_ids=args.deviceid)  # for multi gpu


optimizer = optim.Adam(list(rnet.parameters())+list(BranchFirst.parameters())+list(BranchSecond.parameters())+list(BranchThird.parameters())+list(BranchLast.parameters()), lr=args.lr, weight_decay=args.wd)
# optimizer = optim.Adam(list(rnet.parameters())+list(BranchLast.parameters()), lr=args.lr, weight_decay=args.wd)
# optimizer = optim.Adam(list(BranchFirst.parameters())+list(BranchSecond.parameters())+list(BranchThird.parameters()), lr=args.lr, weight_decay=args.wd)
# optimizer = optim.Adam(list(rnet.parameters())+list(BranchLast.parameters()), lr=args.lr, weight_decay=args.wd)

# configure(args.cv_dir+'/log', flush_secs=5)
lr_scheduler = utils_jump.LrScheduler(optimizer, args.lr, args.lr_decay_ratio, args.epoch_step)
total_tst_time =0
test_cnt = 0
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    lr_scheduler.adjust_learning_rate(epoch)

    train_online_triplet(epoch)
    if epoch%10==0:# and epoch>0:
        tst_t, trn_t = test_all(epoch)
        total_tst_time += tst_t
        test_cnt+=1
    # train_online_triplet(epoch)

print("Average test time = %.6f" % (total_tst_time/test_cnt))

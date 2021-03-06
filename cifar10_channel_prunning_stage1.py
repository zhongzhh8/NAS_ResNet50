import os
# from tensorboard_logger import configure, log_value
import torch
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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
parser = argparse.ArgumentParser(description='BlockDrop Training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')

parser.add_argument('--cv_dir', default='checkpoints_width/cifar10/stage1/wd2_48B/11_20/try', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--binary_bits', type=int, default=48, help='length of hashing binary')
parser.add_argument('--margin', type=float, default=20, help='margin of triplet loss')

parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epoch_step', type=int, default=100, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=600, help='total epochs to run')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='lr *= lr_decay_ratio after epoch_steps')

parser.add_argument('--samples_each_label', type=int, default=20, help='number of samples each label in a batch')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
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
    # BranchFirst.train()
    # BranchSecond.train()
    # BranchThird.train()
    BranchLast.train()

    accum_loss = 0
    batch_time = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(online_train_loader):
        inputs, targets = Variable(inputs), Variable(targets).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()

        time_t = time.time()
        # preds_sample = rnet.forward_last(inputs)
        # out1,out2,out3,out4 = rnet.forward_all(inputs)
        # code1 = BranchFirst.forward(out1)
        # code1 = torch.tanh(code1)
        # code2 = BranchSecond.forward(out2)
        # code2 = torch.tanh(code2)
        # code3 = BranchThird.forward(out3)
        # code3 = torch.tanh(code3)
        # code4 = BranchLast.forward(out4)
        # code4 = torch.tanh(code4)
        out1, out2, out3, out4 = rnet.forward(inputs)
        code4 = BranchLast.forward(out4)

        batch_time.update(time.time() - time_t)

        loss, len_triplets = online_triplet_loss.forward(code4, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accum_loss += loss

    print("epoch: %d, accum_loss: %.6f, average batch time: %.6f" % (epoch, accum_loss, batch_time.avg))

def compute_result(dataloader):
    bs, clses = [], []
    total_time = 0
    batch_time = AverageMeter()
    for img, cls in tqdm.tqdm(dataloader):
        clses.append(cls)
        inputs, targets = Variable(img), Variable(cls).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()

        time_start = time.time()
        with torch.no_grad():
            _, _, _, out4 = rnet.forward(inputs)
            preds = BranchLast.forward(out4)
            # preds = rnet.forward_last(inputs)
            # preds = BranchLast.forward(preds)
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
    for img, cls in tqdm.tqdm(dataloader):
        clses.append(cls)
        inputs, targets = Variable(img), Variable(cls).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()

        time_start = time.time()
        with torch.no_grad():
            # code1,code2,code3,code4 = rnet.forward_all(inputs)
            # code1 = BranchFirst(code1)
            # code2 = BranchSecond(code2)
            # code3 = BranchThird(code3)
            # code4 = BranchLast(code4)
            _, _, _, out4 = rnet.forward(inputs)
            code = BranchLast.forward(out4)
        time_used = time.time() - time_start
        total_time += time_used
        batch_time.update(time_used)
        bs1.append(code.data.cpu())
        bs2.append(code.data.cpu())
        bs3.append(code.data.cpu())
        bs4.append(code.data.cpu())
    return torch.sign(torch.cat(bs1)), torch.sign(torch.cat(bs2)), torch.sign(torch.cat(bs3)), torch.sign(torch.cat(bs4)), torch.cat(clses), total_time, batch_time.avg


def test_all(epoch):

    rnet.eval()
    # BranchFirst.eval()
    # BranchSecond.eval()
    # BranchThird.eval()
    BranchLast.eval()

    tst_code1, tst_code2, tst_code3, tst_code4, tst_label, tst_time, tst_avg_time = compute_result_all(test_loader)
    db_binary, db_label, db_time, db_avg_time = compute_result(database_loader)
    # mAP1 = utils_jump.compute_mAP(db_binary, tst_code1, db_label, tst_label)
    # mAP2 = utils_jump.compute_mAP(db_binary, tst_code2, db_label, tst_label)
    # mAP3 = utils_jump.compute_mAP(db_binary, tst_code3, db_label, tst_label)
    mAP1=mAP2=mAP3=-1
    mAP4 = utils_jump.compute_mAP(db_binary, tst_code4, db_label, tst_label)

    print("epoch: %d, retrieval mAP1: %.6f, retrieval mAP2: %.6f, retrieval mAP3: %.6f, retrieval mAP4: %.6f, " % (epoch, mAP1, mAP2, mAP3, mAP4))


    # mAP4_500 = utils_jump.compute_mAP_topK(db_binary, tst_code4, db_label, tst_label, 500)
    # mAP4_1000 = utils_jump.compute_mAP_topK(db_binary, tst_code4, db_label, tst_label, 1000)
    # mAP4_2000 = utils_jump.compute_mAP_topK(db_binary, tst_code4, db_label, tst_label, 2000)
    # mAP4_5000 = utils_jump.compute_mAP_topK(db_binary, tst_code4, db_label, tst_label, 5000)
    # print("epoch: %d, retrieval mAP4_500: %.6f, retrieval mAP4_1000: %.6f, retrieval mAP4_2000: %.6f, retrieval mAP4_5000: %.6f" % (epoch, mAP4_500, mAP4_1000, mAP4_2000, mAP4_5000))

    # save the model
    rnet_state_dict = rnet.module.state_dict() if args.parallel else rnet.state_dict()
    # branch_first_state = BranchFirst.module.state_dict() if args.parallel else BranchFirst.state_dict()
    # branch_second_state = BranchSecond.module.state_dict() if args.parallel else BranchSecond.state_dict()
    # branch_third_state = BranchThird.module.state_dict() if args.parallel else BranchThird.state_dict()
    branch_last_state = BranchLast.module.state_dict() if args.parallel else BranchLast.state_dict()

    torch.save(rnet_state_dict, args.cv_dir+'/ckpt_E_%d_mAP_%.5f_%.5f_%.5f_%.5f.t7'%(epoch,mAP1,mAP2,mAP3,mAP4))
    # torch.save(branch_first_state, args.cv_dir+'/ckpt_E_%d_mAP_%.5f_%.5f_%.5f_%.5f_branchFirst.t7'%(epoch,mAP1,mAP2,mAP3,mAP4))
    # torch.save(branch_second_state, args.cv_dir+'/ckpt_E_%d_mAP_%.5f_%.5f_%.5f_%.5f_branchSecond.t7'%(epoch,mAP1,mAP2,mAP3,mAP4))
    # torch.save(branch_third_state, args.cv_dir+'/ckpt_E_%d_mAP_%.5f_%.5f_%.5f_%.5f_branchThird.t7'%(epoch,mAP1,mAP2,mAP3,mAP4))
    torch.save(branch_last_state, args.cv_dir+'/ckpt_E_%d_mAP_%.5f_%.5f_%.5f_%.5f_branchLast.t7'%(epoch,mAP1,mAP2,mAP3,mAP4))

    f = open(args.cv_dir + '/a_record.txt', 'a')
    f.write('/ckpt_E_%d_mAP_%.5f_%.5f_%.5f_%.5f.t7' % (epoch, mAP1, mAP2, mAP3, mAP4) + '\n')
    f.close()

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
# online_train_loader = torchdata.DataLoader(trainset32, batch_size=args.batch_size, shuffle=True, num_workers=4)
online_train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=train_batch_sampler, num_workers=4)
# online_test_loader = torch.utils.data.DataLoader(testset, batch_sampler=test_batch_sampler)


online_triplet_loss = OnlineTripletLoss(args.margin,  dataset.AllTripletSelector())


from models import resnet_width
rnet=resnet_width.resnet18_wd2(pretrained=True)
BranchLast=resnet_width.BranchLast(bits=48)

rnet.cuda()
BranchLast.cuda()
start_epoch = 0

optimizer = optim.Adam(list(rnet.parameters())+list(BranchLast.parameters()), lr=args.lr, weight_decay=args.wd)

lr_scheduler = utils_jump.LrScheduler(optimizer, args.lr, args.lr_decay_ratio, args.epoch_step)
total_tst_time =0
test_cnt = 0
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    lr_scheduler.adjust_learning_rate(epoch)
    train_online_triplet(epoch)
    if epoch%10==0 and epoch>0:
        tst_t, trn_t = test_all(epoch)
        total_tst_time += tst_t
        test_cnt+=1


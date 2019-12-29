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
from torch.distributions import Bernoulli
from itertools import combinations
from PIL import Image
import time
import torchvision.transforms as transforms

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
parser = argparse.ArgumentParser(description='BlockDrop Training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
parser.add_argument('--model', default='R34_SUN', help='R<depth>_<dataset> see utils_jump.py for a list of configurations')
parser.add_argument('--data_dir', default='../../../disk2/zenghaien/imagenet/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load rnet+agent from')
parser.add_argument('--pretrained', default=None, help='pretrained policy model checkpoint (from curriculum training)')

parser.add_argument('--cv_dir', default='checkpoints/sun42/11_21/48bits_Ldis+L3_samplr', help='checkpoint directory (models and logs are saved here)')

parser.add_argument('--batch_size', type=int, default=170, help='batch size')
parser.add_argument('--epoch_step', type=int, default=200, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=1000, help='total epochs to run')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='lr *= lr_decay_ratio after epoch_steps')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
# parser.add_argument('--joint', action ='store_true', default=True, help='train both the policy network and the resnet')
parser.add_argument('--penalty', type=float, default=-5, help='gamma: reward for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--alpha2', type=float, default=0.01, help='weighting of regularizer')
parser.add_argument('--binary_bits', type=int, default=48, help='length of hashing binary')
parser.add_argument('--margin', type=float, default=20, help='margin of triplet loss')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils_jump.save_args(__file__, args)

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(torchdata.Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            if line == '':
                break
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(' ')
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)

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
        # losses = F.relu(ap_distances - an_distances + self.margin)
        losses = F.relu(ap_distances - an_distances + 5)

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

        # losses = 0.0
        # ap_distances1 = (code1[triplets4[:, 0]] - code4[triplets4[:, 1]]).pow(2).sum(1)  # .pow(.5)
        # an_distances1 = (code1[triplets4[:, 0]] - code4[triplets4[:, 2]]).pow(2).sum(1)  # .pow(.5)
        # losses += F.relu(ap_distances1 - an_distances1 + self.margin).mean()
        # ap_distances2 = (code2[triplets4[:, 0]] - code4[triplets4[:, 1]]).pow(2).sum(1)  # .pow(.5)
        # an_distances2 = (code2[triplets4[:, 0]] - code4[triplets4[:, 2]]).pow(2).sum(1)  # .pow(.5)
        # losses += F.relu(ap_distances2 - an_distances2 + self.margin).mean()
        # ap_distances3 = (code3[triplets4[:, 0]] - code4[triplets4[:, 1]]).pow(2).sum(1)  # .pow(.5)
        # an_distances3 = (code3[triplets4[:, 0]] - code4[triplets4[:, 2]]).pow(2).sum(1)  # .pow(.5)
        # losses += F.relu(ap_distances3 - an_distances3 + self.margin).mean()

        losses = 0.0
        # ap_distances1 = (code1[triplets1[:, 0]] - code1[triplets1[:, 1]]).pow(2).sum(1)  # .pow(.5)
        # an_distances1 = (code1[triplets1[:, 0]] - code1[triplets1[:, 2]]).pow(2).sum(1)  # .pow(.5)
        # losses += F.relu(ap_distances1 - an_distances1 + self.margin).mean()
        # ap_distances2 = (code2[triplets2[:, 0]] - code2[triplets2[:, 1]]).pow(2).sum(1)  # .pow(.5)
        # an_distances2 = (code2[triplets2[:, 0]] - code2[triplets2[:, 2]]).pow(2).sum(1)  # .pow(.5)
        # losses += F.relu(ap_distances2 - an_distances2 + self.margin).mean()
        # ap_distances3 = (code3[triplets3[:, 0]] - code3[triplets3[:, 1]]).pow(2).sum(1)  # .pow(.5)
        # an_distances3 = (code3[triplets3[:, 0]] - code3[triplets3[:, 2]]).pow(2).sum(1)  # .pow(.5)
        # losses += F.relu(ap_distances3 - an_distances3 + self.margin).mean()
        # ap_distances4 = (code4[triplets4[:, 0]] - code4[triplets4[:, 1]]).pow(2).sum(1)  # .pow(.5)
        # an_distances4 = (code4[triplets4[:, 0]] - code4[triplets4[:, 2]]).pow(2).sum(1)  # .pow(.5)
        # losses += F.relu(ap_distances4 - an_distances4 + self.margin).mean()
        # losses += F.relu(ap_distances4 - an_distances4 + 10).mean()

        losses += (((code1[triplets4[:, 0]] - code4[triplets4[:, 0]]).pow(2).sum(1)).mean())#*0.1
        losses += (((code2[triplets4[:, 0]] - code4[triplets4[:, 0]]).pow(2).sum(1)).mean())#*0.1
        losses += (((code3[triplets4[:, 0]] - code4[triplets4[:, 0]]).pow(2).sum(1)).mean())#*0.1

        return losses, len(triplets4)

class OnlineTripletLoss_regressiom(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss_regressiom, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, code1, code2, code3, code4, code_target, target):

        triplets = self.triplet_selector.get_triplets(code4, target)

        if code4.is_cuda:
            triplets = triplets.cuda()

        losses = 0.0
        ap_distances = (code1[triplets[:, 0]] - code1[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (code1[triplets[:, 0]] - code1[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses += F.relu(ap_distances - an_distances + args.margin).mean()
        ap_distances = (code2[triplets[:, 0]] - code2[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (code2[triplets[:, 0]] - code2[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses += F.relu(ap_distances - an_distances + args.margin).mean()
        ap_distances = (code3[triplets[:, 0]] - code3[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (code3[triplets[:, 0]] - code3[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses += F.relu(ap_distances - an_distances + args.margin).mean()
        ap_distances = (code4[triplets[:, 0]] - code4[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (code4[triplets[:, 0]] - code4[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses += F.relu(ap_distances - an_distances + args.margin).mean()

        losses += (((code1 - code_target).pow(2).sum(1)).mean())
        losses += (((code2 - code_target).pow(2).sum(1)).mean())
        losses += (((code3 - code_target).pow(2).sum(1)).mean())
        losses += (((code4 - code_target).pow(2).sum(1)).mean())

        # losses += (((code1[triplets[:, 0]] - code_target[triplets[:, 0]]).pow(2).sum(1)).mean())
        # losses += (((code2[triplets[:, 0]] - code_target[triplets[:, 0]]).pow(2).sum(1)).mean())
        # losses += (((code3[triplets[:, 0]] - code_target[triplets[:, 0]]).pow(2).sum(1)).mean())
        # losses += (((code4[triplets[:, 0]] - code_target[triplets[:, 0]]).pow(2).sum(1)).mean())

        return losses, len(triplets)

class OnlineTripletLoss_cross_triplet(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss_cross_triplet, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, code1, code2, code3, code4, code_target, target):

        triplets = self.triplet_selector.get_triplets(code4, target)

        if code4.is_cuda:
            triplets = triplets.cuda()

        losses = 0.0
        ap_distances = (code1[triplets[:, 0]] - code_target[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (code1[triplets[:, 0]] - code_target[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses += F.relu(ap_distances - an_distances + 5).mean()
        ap_distances = (code2[triplets[:, 0]] - code_target[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (code2[triplets[:, 0]] - code_target[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses += F.relu(ap_distances - an_distances + 5).mean()
        ap_distances = (code3[triplets[:, 0]] - code_target[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (code3[triplets[:, 0]] - code_target[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses += F.relu(ap_distances - an_distances + 5).mean()
        ap_distances = (code4[triplets[:, 0]] - code_target[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (code4[triplets[:, 0]] - code_target[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses += F.relu(ap_distances - an_distances + 5).mean()

        return losses, len(triplets)

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
    # agent.eval()
    rnet.train()
    BranchFirst.train()
    BranchSecond.train()
    BranchThird.train()
    BranchLast.train()
    rnet_target.eval()
    BranchLast_target.eval()

    accum_loss = 0
    batch_time = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(online_train_loader):

        inputs, targets = Variable(inputs), Variable(targets).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()

        # v_inputs = Variable(inputs.data, volatile=True)
        time_t = time.time()
        out_target = rnet_target.forward_last(inputs)
        code_target = BranchLast_target.forward(out_target)
        code_target = torch.tanh(code_target)

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

        # loss, len_triplets = online_triplet_loss.forward(code1, code2, code3, code4, targets)
        # loss, len_triplets = online_triplet_loss.forward(code4, targets)
        loss, len_triplets = online_triplet_loss.forward(code1, code2, code3, code4, code_target, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accum_loss += loss

    # logger.add_scalar('accum_loss', accum_loss, epoch)
    print("epoch: %d, accum_loss: %.6f, average batch time: %.6f" % (epoch, accum_loss, batch_time.avg))

def compute_result(dataloader):
    bs, clses = [], []
    # rewards, policies = [], []
    rnet.eval()
    total_time = 0
    batch_time = AverageMeter()
    for img, cls in dataloader:
        clses.append(cls)
        inputs, targets = Variable(img), Variable(cls).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()

        time_start = time.time()
        # preds = rnet.forward(inputs, policy)
        with torch.no_grad():
            preds = rnet_target.forward_last(inputs)
            preds = BranchLast_target.forward(preds)
        # preds = F.tanh(preds)
        time_used = time.time() - time_start
        total_time += time_used
        # print(preds.size())
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
        # print(code1.size())
        # print(code2.size())
        # print(code3.size())
        # print(code4.size())
        time_used = time.time() - time_start
        total_time += time_used
        batch_time.update(time_used)
        bs1.append(code1.data.cpu())
        bs2.append(code2.data.cpu())
        bs3.append(code3.data.cpu())
        bs4.append(code4.data.cpu())
    return torch.sign(torch.cat(bs1)), torch.sign(torch.cat(bs2)), torch.sign(torch.cat(bs3)), torch.sign(torch.cat(bs4)), torch.cat(clses), total_time, batch_time.avg

def test_all(epoch):

    # agent.eval()
    rnet.eval()
    BranchFirst.eval()
    BranchSecond.eval()
    BranchThird.eval()
    BranchLast.eval()
    rnet_target.eval()
    BranchLast_target.eval()

    tst_code1, tst_code2, tst_code3, tst_code4, tst_label, tst_time, tst_avg_time = compute_result_all(test_loader_256)
    # tst_code4, tst_label, tst_time, tst_avg_time = compute_result(test_loader_256)
    db_binary, db_label, db_time, db_avg_time = compute_result(db_loader_256)
    # tst_code4, tst_label, tst_time, tst_avg_time = compute_result(testloader)
    # print('total test time = %.6f, average test time = %.6f ,total train time = %.6f, average train time = %.6f'%(tst_time, tst_avg_time ,db_time, db_avg_time))

    mAP1 = utils_jump.compute_mAP(db_binary, tst_code1, db_label, tst_label)
    mAP2 = utils_jump.compute_mAP(db_binary, tst_code2, db_label, tst_label)
    mAP3 = utils_jump.compute_mAP(db_binary, tst_code3, db_label, tst_label)
    mAP4 = utils_jump.compute_mAP(db_binary, tst_code4, db_label, tst_label)
    print("epoch: %d, retrieval mAP1: %.6f, retrieval mAP2: %.6f, retrieval mAP3: %.6f, retrieval mAP4: %.6f, " % (epoch, mAP1, mAP2, mAP3, mAP4))

    # mAP1_500 = utils_jump.compute_mAP_topK(db_binary, tst_code1, db_label, tst_label, 500)
    # mAP1_1000 = utils_jump.compute_mAP_topK(db_binary, tst_code1, db_label, tst_label, 1000)
    # mAP1_2000 = utils_jump.compute_mAP_topK(db_binary, tst_code1, db_label, tst_label, 2000)
    # mAP1_5000 = utils_jump.compute_mAP_topK(db_binary, tst_code1, db_label, tst_label, 5000)
    # print("epoch: %d, retrieval mAP1_500: %.6f, retrieval mAP1_1000: %.6f, retrieval mAP1_2000: %.6f, retrieval mAP1_5000: %.6f, " % (
    #         epoch, mAP1_500, mAP1_1000, mAP1_2000, mAP1_5000))
    #
    # mAP2_500 = utils_jump.compute_mAP_topK(db_binary, tst_code2, db_label, tst_label, 500)
    # mAP2_1000 = utils_jump.compute_mAP_topK(db_binary, tst_code2, db_label, tst_label, 1000)
    # mAP2_2000 = utils_jump.compute_mAP_topK(db_binary, tst_code2, db_label, tst_label, 2000)
    # mAP2_5000 = utils_jump.compute_mAP_topK(db_binary, tst_code2, db_label, tst_label, 5000)
    # print("epoch: %d, retrieval mAP2_500: %.6f, retrieval mAP2_1000: %.6f, retrieval mAP2_2000: %.6f, retrieval mAP2_5000: %.6f, " % (
    #         epoch, mAP2_500, mAP2_1000, mAP2_2000, mAP2_5000))
    #
    # mAP3_500 = utils_jump.compute_mAP_topK(db_binary, tst_code3, db_label, tst_label, 500)
    # mAP3_1000 = utils_jump.compute_mAP_topK(db_binary, tst_code3, db_label, tst_label, 1000)
    # mAP3_2000 = utils_jump.compute_mAP_topK(db_binary, tst_code3, db_label, tst_label, 2000)
    # mAP3_5000 = utils_jump.compute_mAP_topK(db_binary, tst_code3, db_label, tst_label, 5000)
    # print("epoch: %d, retrieval mAP3_500: %.6f, retrieval mAP3_1000: %.6f, retrieval mAP3_2000: %.6f, retrieval mAP3_5000: %.6f, " % (
    #         epoch, mAP3_500, mAP3_1000, mAP3_2000, mAP3_5000))
    #
    # mAP4_500 = utils_jump.compute_mAP_topK(db_binary, tst_code4, db_label, tst_label, 500)
    # mAP4_1000 = utils_jump.compute_mAP_topK(db_binary, tst_code4, db_label, tst_label, 1000)
    # mAP4_2000 = utils_jump.compute_mAP_topK(db_binary, tst_code4, db_label, tst_label, 2000)
    # mAP4_5000 = utils_jump.compute_mAP_topK(db_binary, tst_code4, db_label, tst_label, 5000)
    # print("epoch: %d, retrieval mAP4_500: %.6f, retrieval mAP4_1000: %.6f, retrieval mAP4_2000: %.6f, retrieval mAP4_5000: %.6f, " % (
    #         epoch, mAP4_500, mAP4_1000, mAP4_2000, mAP4_5000))

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

    f = open(args.cv_dir + '/a_record.txt', 'a')
    f.write('/ckpt_E_%d_mAP_%.5f_%.5f_%.5f_%.5f.t7' % (epoch, mAP1, mAP2, mAP3, mAP4) + '\n')
    f.close()

    return tst_time,db_time

def test(epoch):

    rnet.eval()
    BranchFirst.eval()
    BranchSecond.eval()
    BranchThird.eval()
    BranchLast.eval()

    tst_binary, tst_label, tst_time, tst_avg_time = compute_result(test_loader_256)
    trn_binary, trn_label, trn_time, trn_avg_time = compute_result(train_loader_256)
    # print('total test time = %.6f, average test time = %.6f ,total train time = %.6f, average train time = %.6f'%(tst_time, tst_avg_time ,trn_time, trn_avg_time))

    mAP = utils_jump.compute_mAP(trn_binary, tst_binary, trn_label, tst_label)
    # print(f'[{epoch}] retrieval mAP: {mAP:.4f}')
    print("epoch: %d, retrieval mAP: %.6f" %(epoch, mAP))

    # save the model
    # agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()
    rnet_state_dict = rnet.module.state_dict() if args.parallel else rnet.state_dict()
    branch_last_state = BranchLast.module.state_dict() if args.parallel else BranchLast.state_dict()

    torch.save(rnet_state_dict, args.cv_dir+'/ckpt_E_%d_mAP_%.5f.t7'%(epoch, mAP))
    torch.save(branch_last_state, args.cv_dir+'/ckpt_E_%d_mAP_%.5f_branchLast.t7'%(epoch, mAP))


    return tst_time,trn_time


#--------------------------------------------------------------------------------------------------------#
root_dir = './data/SUN42/'
train_dir = root_dir+'_sun42_train_500.txt'
db_dir = root_dir+'_sun42_database.txt'
test_dir = root_dir+'_test256_it_singleFolder_32b.txt'
transform_train_256 = transforms.Compose([transforms.Resize((256, 256)),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
transform_test_256 = transforms.Compose([transforms.Resize((256, 256)),transforms.CenterCrop(224),transforms.ToTensor()])

trainset256 = MyDataset(txt=train_dir, transform=transform_train_256)
testset256 = MyDataset(txt=test_dir, transform=transform_test_256)
database256 = MyDataset(txt=db_dir, transform=transform_test_256)

# train_batch_sampler = dataset.BalancedBatchSampler_SUN_SF_RPT(trainset256, n_classes=42, n_samples=6)

train_loader_256 = torchdata.DataLoader(trainset256, batch_size=args.batch_size,shuffle=True)
test_loader_256 = torchdata.DataLoader(testset256, batch_size=args.batch_size,shuffle=False)
db_loader_256 = torchdata.DataLoader(database256, batch_size=args.batch_size,shuffle=False)

# online_train_loader = torch.utils.data.DataLoader(trainset256, batch_sampler=train_batch_sampler, num_workers=4)
online_train_loader = torch.utils.data.DataLoader(trainset256, batch_size=args.batch_size, shuffle=True, num_workers=4)

# online_triplet_loss = OnlineTripletLoss(args.margin, dataset.RandomNegativeTripletSelector(args.margin))
# online_triplet_loss = OnlineTripletLoss_jump(args.margin, dataset.SemihardNegativeTripletSelector(args.margin))
# online_triplet_loss = OnlineTripletLoss(args.margin, dataset.HardestNegativeTripletSelector(args.margin))
# online_triplet_loss = OnlineTripletLoss_jump(args.margin, dataset.AllTripletSelector())
# online_triplet_loss = OnlineTripletLoss(args.margin, dataset.AllTripletSelector())
online_triplet_loss = OnlineTripletLoss_regressiom(args.margin, dataset.AllTripletSelector())
# online_triplet_loss = OnlineTripletLoss_cross_triplet(args.margin, dataset.AllTripletSelector())


from models import resnet_standard
# rnet = resnet_standard.resnet34(pretrained=True)

# rnet = resnet_standard.resnet18(pretrained=True)
rnet = resnet_standard.resnet18(pretrained=True)
BranchFirst = resnet_standard.BranchFirst(inplanes=64, expansion=1, bits=48)
BranchSecond = resnet_standard.BranchSecond(inplanes=128, expansion=1, bits=48)
BranchThird = resnet_standard.BranchThird(inplanes=256, expansion=1, bits=48)
BranchLast = resnet_standard.BranchLast(inplanes=512, expansion=1, bits=48)

rnet_target = resnet_standard.resnet18(pretrained=False)
BranchLast_target = resnet_standard.BranchLast(inplanes=512, expansion=1, bits=48)
# rnet_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_12B/ckpt_E_90_mAP_-1.00000_-1.00000_-1.00000_0.79667.t7'
# rnet_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_24B/ckpt_E_140_mAP_-1.00000_-1.00000_-1.00000_0.82413.t7'
# rnet_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_32B/ckpt_E_110_mAP_-1.00000_-1.00000_-1.00000_0.83605.t7'
rnet_checkpoint = 'pretrained/sun42/R18_48B/ckpt_E_140_mAP_-1.00000_-1.00000_-1.00000_0.84578.t7'
rnet_checkpoint = torch.load(rnet_checkpoint)
# rnet.load_state_dict(rnet_checkpoint)
rnet_target.load_state_dict(rnet_checkpoint)

# rnet_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_48B/ckpt_E_240_mAP_0.79717_0.82467_0.84732_0.85436.t7'
# rnet_checkpoint = torch.load(rnet_checkpoint)
# rnet.load_state_dict(rnet_checkpoint)

# b1_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_12B/ckpt_E_110_mAP_0.68477_0.71214_0.74094_0.79667_branchFirst.t7'
# b1_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_24B/ckpt_E_130_mAP_0.74242_0.77125_0.78684_0.82413_branchFirst.t7'
# b1_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_32B/ckpt_E_90_mAP_0.76722_0.79031_0.80456_0.83518_branchFirst.t7'
# b1_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_48B/ckpt_E_110_mAP_0.77566_0.79984_0.81713_0.84563_branchFirst.t7'
# b1_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_48B/ckpt_E_240_mAP_0.79717_0.82467_0.84732_0.85436_branchFirst.t7'
# b1_checkpoint = torch.load(b1_checkpoint)
# BranchFirst.load_state_dict(b1_checkpoint)

# b2_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_12B/ckpt_E_100_mAP_0.68329_0.71335_0.74049_0.79667_branchSecond.t7'
# b2_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_24B/ckpt_E_130_mAP_0.74242_0.77125_0.78684_0.82413_branchSecond.t7'
# b2_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_32B/ckpt_E_100_mAP_0.76697_0.79310_0.80427_0.83518_branchSecond.t7'
# b2_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_48B/ckpt_E_110_mAP_0.77566_0.79984_0.81713_0.84563_branchSecond.t7'
# b2_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_48B/ckpt_E_240_mAP_0.79717_0.82467_0.84732_0.85436_branchSecond.t7'
# b2_checkpoint = torch.load(b2_checkpoint)
# BranchSecond.load_state_dict(b2_checkpoint)

# b3_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_12B/ckpt_E_90_mAP_0.68060_0.70894_0.74133_0.79667_branchThird.t7'
# b3_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_24B/ckpt_E_90_mAP_0.73949_0.77042_0.78908_0.82413_branchThird.t7'
# b3_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_32B/ckpt_E_90_mAP_0.76722_0.79031_0.80456_0.83518_branchThird.t7'
# b3_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_48B/ckpt_E_110_mAP_0.77566_0.79984_0.81713_0.84563_branchThird.t7'
# b3_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_48B/ckpt_E_240_mAP_0.79717_0.82467_0.84732_0.85436_branchThird.t7'
# b3_checkpoint = torch.load(b3_checkpoint)
# BranchThird.load_state_dict(b3_checkpoint)

# b4_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_12B/ckpt_E_90_mAP_-1.00000_-1.00000_-1.00000_0.79667_branchLast.t7'
# b4_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_24B/ckpt_E_140_mAP_-1.00000_-1.00000_-1.00000_0.82413_branchLast.t7'
# b4_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_32B/ckpt_E_110_mAP_-1.00000_-1.00000_-1.00000_0.83605_branchLast.t7'
b4_checkpoint = 'pretrained/sun42/R18_48B/ckpt_E_140_mAP_-1.00000_-1.00000_-1.00000_0.84578_branchLast.t7'
b4_checkpoint = torch.load(b4_checkpoint)
# BranchLast.load_state_dict(b4_checkpoint)
BranchLast_target.load_state_dict(b4_checkpoint)
# b4_checkpoint = '../../../disk2/zenghaien/ckt/pretrained/sun42/R18_48B/ckpt_E_240_mAP_0.79717_0.82467_0.84732_0.85436_branchLast.t7'
# b4_checkpoint = torch.load(b4_checkpoint)
# BranchLast.load_state_dict(b4_checkpoint)


rnet_checkpoint ='/home/disk1/azzh/PycharmProject/resnetKDs/resnetKD0/checkpoints/sun42/11_20/48bits_Ldis_step250/ckpt_E_150_mAP_0.75376_0.81179_0.83969_0.84950.t7'
b1_checkpoint='/home/disk1/azzh/PycharmProject/resnetKDs/resnetKD0/checkpoints/sun42/11_20/48bits_Ldis_step250/ckpt_E_150_mAP_0.75376_0.81179_0.83969_0.84950_branchFirst.t7'
b2_checkpoint='/home/disk1/azzh/PycharmProject/resnetKDs/resnetKD0/checkpoints/sun42/11_20/48bits_Ldis_step250/ckpt_E_150_mAP_0.75376_0.81179_0.83969_0.84950_branchSecond.t7'
b3_checkpoint='/home/disk1/azzh/PycharmProject/resnetKDs/resnetKD0/checkpoints/sun42/11_20/48bits_Ldis_step250/ckpt_E_150_mAP_0.75376_0.81179_0.83969_0.84950_branchThird.t7'
b4_checkpoint='/home/disk1/azzh/PycharmProject/resnetKDs/resnetKD0/checkpoints/sun42/11_20/48bits_Ldis_step250/ckpt_E_150_mAP_0.75376_0.81179_0.83969_0.84950_branchLast.t7'
rnet_checkpoint = torch.load(rnet_checkpoint)
rnet.load_state_dict(rnet_checkpoint)
b1_checkpoint = torch.load(b1_checkpoint)
BranchFirst.load_state_dict(b1_checkpoint)
b2_checkpoint = torch.load(b2_checkpoint)
BranchSecond.load_state_dict(b2_checkpoint)
b3_checkpoint = torch.load(b3_checkpoint)
BranchThird.load_state_dict(b3_checkpoint)
b4_checkpoint = torch.load(b4_checkpoint)
BranchLast.load_state_dict(b4_checkpoint)



start_epoch = 0

rnet.cuda()
BranchFirst.cuda()
BranchSecond.cuda()
BranchThird.cuda()
BranchLast.cuda()
rnet_target.cuda()
BranchLast_target.cuda()

optimizer = optim.Adam(list(rnet.parameters())+list(BranchFirst.parameters())+list(BranchSecond.parameters())+list(BranchThird.parameters())+list(BranchLast.parameters()), lr=args.lr, weight_decay=args.wd)
# optimizer = optim.Adam(list(BranchFirst.parameters())+list(BranchSecond.parameters())+list(BranchThird.parameters()), lr=args.lr, weight_decay=args.wd)
# optimizer = optim.Adam(list(rnet.parameters())+list(BranchLast.parameters()), lr=args.lr, weight_decay=args.wd)

# configure(args.cv_dir+'/log', flush_secs=5)

lr_scheduler = utils_jump.LrScheduler(optimizer, args.lr, args.lr_decay_ratio, args.epoch_step)
total_tst_time =0
test_cnt = 0
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    lr_scheduler.adjust_learning_rate(epoch)

    # tst_t, trn_t = test_all(epoch)
    # break

    train_online_triplet(epoch)
    if epoch%10==0 and epoch>0:
        tst_t, trn_t = test_all(epoch)
        total_tst_time += tst_t
        test_cnt+=1
    # train_online_triplet(epoch)

print("Average test time = %.6f" % (total_tst_time/test_cnt))

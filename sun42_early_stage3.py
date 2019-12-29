import os
import torch
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import tqdm
import utils_jump
import dataset
import torch.optim as optim
import time
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser(description='BlockDrop Training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')

parser.add_argument('--deviceid', type=list, default=[3], help='')

parser.add_argument('--cv_dir', default='checkpoints_early/distill/sun42/r50_48B/12_23/default', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--binary_bits', type=int, default=48, help='length of hashing binary')
# parser.add_argument('--margin', type=float, default=20, help='margin of triplet loss')


parser.add_argument('--lr_step', type=int, default=50, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=110, help='total epochs to run')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='lr *= lr_decay_ratio after epoch_steps')

# parser.add_argument('--samples_each_label', type=int, default=10, help='number of samples each label in a batch')
parser.add_argument('--batch_size', type=int, default=30, help='batch size')
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
        losses = 0.0
        triplets = self.triplet_selector.get_triplets(code4, target)
        triplets = triplets.to(device)
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

        return losses, -1


class EuclideanDistance(nn.Module):
    """
    就是欧氏距离
    """
    def __init__(self):
        super(EuclideanDistance, self).__init__()

    def forward(self, code1, code2, code3, code4, code_target, target):
        losses = 0.0
        losses += (((code1 - code_target).pow(2).sum(1)).mean())
        losses += (((code2 - code_target).pow(2).sum(1)).mean())
        losses += (((code3 - code_target).pow(2).sum(1)).mean())
        losses += (((code4 - code_target).pow(2).sum(1)).mean())
        return losses, -1

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
            _,_,_,out = rnet_target(inputs)
            preds = BranchLast_target(out)
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
        inputs, targets = Variable(img).to(device), Variable(cls).to(device)

        time_start = time.time()
        with torch.no_grad():
            code1,code2,code3,code4 = rnet(inputs)
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
    total_mAP=0
    f_test = open(args.cv_dir + '/a_test_record.txt', 'a')
    for width_mult in sorted(width_mult_list, reverse=True):  # 1x,0.75x,0.5x,0.25x依次训练，用1x的soft target蒸馏训练其他
        if width_mult!=1.0: #只训练1.0x
            continue
        def fn(m):
            setattr(m, 'width_mult', width_mult)  # width_mult在SlimmableConv2d和SwitchableBatchNorm2d里有
        rnet.apply(fn)
        BranchFirst.apply(fn)
        BranchSecond.apply(fn)
        BranchThird.apply(fn)
        BranchLast.apply(fn)
        rnet.eval()
        BranchFirst.eval()
        BranchSecond.eval()
        BranchThird.eval()
        BranchLast.eval()

        tst_code1, tst_code2, tst_code3, tst_code4, tst_label, tst_time, tst_avg_time = compute_result_all(test_loader)
        mAP1 = utils_jump.compute_mAP(db_binary, tst_code1, db_label, tst_label)
        mAP2 = utils_jump.compute_mAP(db_binary, tst_code2, db_label, tst_label)
        mAP3 = utils_jump.compute_mAP(db_binary, tst_code3, db_label, tst_label)
        mAP4 = utils_jump.compute_mAP(db_binary, tst_code4, db_label, tst_label)
        total_mAP+=mAP1+mAP2+mAP3+mAP4
        avg_mAP=total_mAP/16
        print("width %.2f epoch: %d, retrieval mAP1: %.6f, retrieval mAP2: %.6f, retrieval mAP3: %.6f, retrieval mAP4: %.6f, " % (width_mult,epoch, mAP1, mAP2, mAP3, mAP4))

        if width_mult==1.0 and epoch>0:
            torch.save({
                'rnet_state_dict': rnet.state_dict(),
                'branch_first_state_dict': BranchFirst.state_dict(),
                'branch_second_state_dict': BranchSecond.state_dict(),
                'branch_third_state_dict': BranchThird.state_dict(),
                'branch_last_state_dict': BranchLast.state_dict(),
            }, args.cv_dir+'/E_%03d_mAP_%.5f_%.5f_%.5f_%.5f.pth'%(epoch,mAP1,mAP2,mAP3,mAP4))


        f_test.write('width %.2f E_%d_mAP_%.5f_%.5f_%.5f_%.5f.t7' % (width_mult,epoch, mAP1, mAP2, mAP3, mAP4) + '\n')
    f_test.write('E_%d_total_mAP=%.5f_avg_mAP=%.5f' % (epoch, total_mAP,avg_mAP) + '\n')
    f_test.close()

def Load_data():
    root_dir = './data/SUN42/'
    train_dir = root_dir + '_sun42_train_500.txt'
    db_dir = root_dir + '_sun42_database.txt'
    test_dir = root_dir + '_test256_it_singleFolder_32b.txt'
    transform_train_256 = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    transform_test_256 = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor()])

    trainset256 = MyDataset(txt=train_dir, transform=transform_train_256)
    testset256 = MyDataset(txt=test_dir, transform=transform_test_256)
    database256 = MyDataset(txt=db_dir, transform=transform_test_256)

    # train_batch_sampler = dataset.BalancedBatchSampler_SUN_SF_RPT(trainset256, n_classes=42, n_samples=args.samples_each_label)

    train_loader_256 = torchdata.DataLoader(trainset256, batch_size=args.batch_size, shuffle=True)
    test_loader_256 = torchdata.DataLoader(testset256, batch_size=args.batch_size, shuffle=False)
    db_loader_256 = torchdata.DataLoader(database256, batch_size=args.batch_size, shuffle=False)

    # online_train_loader = torch.utils.data.DataLoader(trainset256, batch_sampler=train_batch_sampler, num_workers=4)
    online_train_loader = torch.utils.data.DataLoader(trainset256, batch_size=args.batch_size, shuffle=True,num_workers=4)
    return train_loader_256,test_loader_256,db_loader_256,online_train_loader

def Load_checkpoint():
    # target加载teacher模型
    target_checkpoint = torch.load(
        # '/data1/azzh/PycharmProject/resnetKD0/checkpoints_slimmable/teachers/sun42/r50_12B/11_28/all4width_batchsize100/E_110_mAP_-1.00000_-1.00000_-1.00000_0.83587.pth',
        # '/data1/azzh/PycharmProject/resnetKD0/checkpoints_slimmable/teachers/sun42/r50_24B/11_29/all4width_batchsize100/E_120_mAP_-1.00000_-1.00000_-1.00000_0.85736.pth',
        # '/data1/azzh/PycharmProject/resnetKD0/checkpoints_slimmable/teachers/sun42/r50_32B/11_29/all4width_batchsize100/E_110_mAP_-1.00000_-1.00000_-1.00000_0.86297.pth',
        '/data1/azzh/PycharmProject/resnetKD0/checkpoints_slimmable/teachers/sun42/r50_48B/11_26/all4width/E_110_mAP_-1.00000_-1.00000_-1.00000_0.87103.pth',
        map_location='cpu')
    rnet_target.load_state_dict(target_checkpoint['rnet_state_dict'])
    BranchLast_target.load_state_dict(target_checkpoint['branch_last_state_dict'])

    rnet.load_state_dict(target_checkpoint['rnet_state_dict'])
    BranchLast.load_state_dict(target_checkpoint['branch_last_state_dict'])

    # #尝试加载预训练参数进branch1,2的bottleneck中
    # rnet_dict=target_checkpoint['rnet_state_dict']
    # b4_dict=target_checkpoint['branch_last_state_dict']
    #
    # b1_dict=BranchFirst.state_dict()
    # b2_dict=BranchSecond.state_dict()
    # b3_dict = BranchThird.state_dict()
    # new_b1_dict={}
    # new_b2_dict = {}
    #
    # for (k,v) in rnet_dict.items():#先把rnet与b1,b2重合的键值对保存起来
    #     if 'features.8' in k:
    #         new_b1_dict[k] = v
    #     if 'features.14' in k:
    #         new_b1_dict[k] = v
    #         new_b2_dict[k] = v
    #
    # for (k, v) in b4_dict.items():#再把b4与b1,b2重合的键值对保存起来
    #     new_b1_dict[k] = v
    #     new_b2_dict[k] = v
    #
    # for ((k, v),(new_k,new_v)) in zip(b1_dict.items(),new_b1_dict.items()): #替换value
    #     b1_dict[k]=new_v
    # for ((k, v),(new_k,new_v)) in zip(b2_dict.items(),new_b2_dict.items()):#替换value
    #     b2_dict[k]=new_v
    # b3_dict.update(b4_dict)
    #
    # BranchFirst.load_state_dict(b1_dict)
    # BranchSecond.load_state_dict(b2_dict)
    # BranchThird.load_state_dict(b3_dict)



#--------------------------------------------------------------------------------------------------------#
device = torch.device("cuda:" + str(args.deviceid[0]) if torch.cuda.is_available() else "cpu")  # device configuration
train_loader,test_loader,database_loader,online_train_loader=Load_data()

LossFunction = EuclideanDistance()
# LossFunction = OnlineTripletLoss_regressiom(args.margin, dataset.AllTripletSelector())

from models import s_resnet
rnet_target=s_resnet.Model()
BranchLast_target=s_resnet.BranchLast(bits=args.binary_bits)

rnet=s_resnet.Model()
BranchFirst=s_resnet.BranchFirst(bits=args.binary_bits)
BranchSecond=s_resnet.BranchSecond(bits=args.binary_bits)
BranchThird=s_resnet.BranchThird(bits=args.binary_bits)
BranchLast=s_resnet.BranchLast(bits=args.binary_bits)

rnet_target.to(device)
rnet_target = torch.nn.DataParallel(rnet_target, device_ids=args.deviceid)
BranchLast_target.to(device)
BranchLast_target=torch.nn.DataParallel(BranchLast_target, device_ids=args.deviceid)
rnet.to(device)
rnet = torch.nn.DataParallel(rnet, device_ids=args.deviceid)
BranchFirst.to(device)
BranchFirst=torch.nn.DataParallel(BranchFirst, device_ids=args.deviceid)
BranchSecond.to(device)
BranchSecond=torch.nn.DataParallel(BranchSecond, device_ids=args.deviceid)
BranchThird.to(device)
BranchThird=torch.nn.DataParallel(BranchThird, device_ids=args.deviceid)
BranchLast.to(device)
BranchLast=torch.nn.DataParallel(BranchLast, device_ids=args.deviceid)

Load_checkpoint()

width_mult_list=[0.25, 0.5, 0.75, 1.0]
start_epoch = 0
optimizer = optim.Adam(list(rnet.parameters())+list(BranchFirst.parameters())+list(BranchSecond.parameters())+list(BranchThird.parameters())+list(BranchLast.parameters()), lr=args.lr, weight_decay=args.wd)
lr_scheduler = utils_jump.LrScheduler(optimizer, args.lr, args.lr_decay_ratio, args.lr_step)
total_tst_time =0
test_cnt = 0


f_train = open(args.cv_dir + '/a_train_record.txt', 'a')
rnet_target.eval()
BranchLast_target.eval()
db_binary, db_label, db_time, db_avg_time = compute_result(database_loader)

# print('check teacher')
# test_all(-1)

for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    lr_scheduler.adjust_learning_rate(epoch)
    for width_mult in sorted(width_mult_list, reverse=True):  # 1x,0.75x,0.5x,0.25x依次训练，用1x的soft target蒸馏训练其他
        if width_mult!=1.0: #只训练1.0x
            continue
        def fn(m):
            setattr(m, 'width_mult', width_mult) #width_mult在SlimmableConv2d和SwitchableBatchNorm2d里有
        rnet.apply(fn)
        BranchFirst.apply(fn)
        BranchSecond.apply(fn)
        BranchThird.apply(fn)
        BranchLast.apply(fn)
        #target模型不需要改宽度，保持1x即可
        rnet.train()
        BranchFirst.train()
        BranchSecond.train()
        BranchThird.train()
        BranchLast.train()

        accum_loss = 0
        batch_time = AverageMeter()
        for batch_idx, (inputs, targets) in enumerate(online_train_loader):
            inputs, targets = Variable(inputs).to(device), Variable(targets).to(device)
            time_t = time.time()
            out1,out2,out3,out4=rnet(inputs)
            code1 = BranchFirst(out1)
            code2 = BranchSecond(out2)
            code3 = BranchThird(out3)
            code4 = BranchLast(out4)

            with torch.no_grad():
                _, _, _, out_target = rnet_target(inputs)
                code_target = BranchLast_target(out_target)

            batch_time.update(time.time() - time_t)

            loss, len_triplets = LossFunction.forward(code1,code2,code3,code4,code_target, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accum_loss += loss

        print("%.2fx epoch: %d, accum_loss: %.6f, average batch time: %.6f" % (width_mult,epoch, accum_loss, batch_time.avg))
        f_train.write("%.2fx epoch: %d, accum_loss: %.6f, average batch time: %.6f" % (width_mult, epoch, accum_loss, batch_time.avg) + '\n')

    if epoch % 10 == 0 and epoch > 0:
        test_all(epoch)


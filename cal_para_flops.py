import torch


import argparse
parser = argparse.ArgumentParser(description='BlockDrop Training')
parser.add_argument('--deviceid', type=list, default=[0], help='')
args = parser.parse_args()

from models import s_resnet
rnet=s_resnet.Model()
BranchFirst=s_resnet.BranchFirst(bits=48)
BranchSecond=s_resnet.BranchSecond(bits=48)
BranchThird=s_resnet.BranchThird(bits=48)
BranchLast=s_resnet.BranchLast(bits=48)

# device = torch.device("cuda:" + str(args.deviceid[0]) if torch.cuda.is_available() else "cpu")
# rnet.to(device)
# rnet = torch.nn.DataParallel(rnet, device_ids=args.deviceid)
# BranchFirst.to(device)
# BranchFirst=torch.nn.DataParallel(BranchFirst, device_ids=args.deviceid)
# BranchSecond.to(device)
# BranchSecond=torch.nn.DataParallel(BranchSecond, device_ids=args.deviceid)
# BranchThird.to(device)
# BranchThird=torch.nn.DataParallel(BranchThird, device_ids=args.deviceid)
# BranchLast.to(device)
# BranchLast=torch.nn.DataParallel(BranchLast, device_ids=args.deviceid)


from thop import profile


def fn(m):
    setattr(m, 'width_mult', 0.5)  # width_mult在SlimmableConv2d和SwitchableBatchNorm2d里有
rnet.apply(fn)

input = torch.randn(1, 3, 224, 224) #模型输入的形状,batch_size=1
flops, params = profile(rnet, inputs=(input, ))
print(flops/1e9,params/1e6) #flops单位G，para单位M

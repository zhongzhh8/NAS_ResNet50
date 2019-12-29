import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import time
from functools import wraps

# Save the training script and all the arguments to a file so that you
# don't feel like an idiot later when you can't replicate results
import shutil
def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def performance_stats(policies, rewards):

    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)
    # accuracy = torch.cat(matches, 0).mean()

    reward = rewards.mean()
    sparsity = policies.sum(1).mean()
    variance = policies.sum(1).std()

    policy_set = [p.cpu().numpy().astype(np.int).astype(np.str) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    # return accuracy, reward, sparsity, variance, policy_set
    return reward, sparsity, variance, policy_set

class LrScheduler:
    def __init__(self, optimizer, base_lr, lr_decay_ratio, epoch_step):
        self.base_lr = base_lr
        self.lr_decay_ratio = lr_decay_ratio
        self.epoch_step = epoch_step
        self.optimizer = optimizer

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.base_lr * (self.lr_decay_ratio ** (epoch // self.epoch_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if epoch%self.epoch_step==0:
                print( '# setting learning_rate to %.2E'%lr)


# load model weights trained using scripts from https://github.com/felixgwu/img_classification_pk_pytorch OR
# from torchvision models into our flattened resnets
def load_weights_to_flatresnet(source_model, target_model):

    # compatibility for nn.Modules + checkpoints
    # if hasattr(source_model, 'rnet_state_dict'):
    #     source_model = {'rnet_state_dict': source_model.state_dict()}
    # source_state = source_model['rnet_state_dict']
    source_state = source_model
    target_state = target_model.state_dict()

    # remove the module. prefix if it exists (thanks nn.DataParallel)
    if source_state.keys()[0].startswith('module.'):
        source_state = {k[7:]:v for k,v in source_state.items()}


    common = set(['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var','fc.weight', 'fc.bias'])
    for key in source_state.keys():

        if key in common:
            target_state[key] = source_state[key]
            continue

        if 'downsample' in key:
            layer, num, item = re.match('layer(\d+).*\.(\d+)\.(.*)', key).groups()
            translated = 'ds.%s.%s.%s'%(int(layer)-1, num, item)
        else:
            layer, item = re.match('layer(\d+)\.(.*)', key).groups()
            translated = 'blocks.%s.%s'%(int(layer)-1, item)


        if translated in target_state.keys():
            target_state[translated] = source_state[key]
        else:
            print( translated, 'block missing')

    target_model.load_state_dict(target_state)
    return target_model

def load_checkpoint(rnet, agent, load):
    if load=='nil':
        return None

    checkpoint = torch.load(load)
    if 'resnet' in checkpoint:
        rnet.load_state_dict(checkpoint['resnet'])
        print ('loaded resnet from', os.path.basename(load))
    if 'agent' in checkpoint:
        agent.load_state_dict(checkpoint['agent'])
        print ('loaded agent from', os.path.basename(load))
    # backward compatibility (some old checkpoints)
    if 'net' in checkpoint:
        checkpoint['net'] = {k:v for k,v in checkpoint['net'].items() if 'features.fc' not in k}
        agent.load_state_dict(checkpoint['net'])
        print ('loaded agent from', os.path.basename(load))


def get_transforms(rnet, dset):

    # Only the R32 pretrained model subtracts the mean, sorry :(
    if dset=='C10' and rnet=='R32':
        mean = [x/255.0 for x in [125.3, 123.0, 113.9]]
        std = [x/255.0 for x in [63.0, 62.1, 66.7]]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

    elif dset=='C100' or dset=='C10' and rnet!='R32':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])

    elif dset=='ImgNet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            # transforms.Resize((256,256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        transform_test = transforms.Compose([
            # transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

    elif dset=='SUN':
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    elif dset=='SUNhalf':
        transform_train = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomCrop(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
        ])

    return transform_train, transform_test

# Pick from the datasets available and the hundreds of models we have lying around depending on the requirements.
def get_dataset(model, root='data/'):

    rnet, dset = model.split('_')
    transform_train, transform_test = get_transforms(rnet, dset)

    if dset=='C10':
        trainset = torchdata.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchdata.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    elif dset=='C100':
        trainset = torchdata.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchdata.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    elif dset=='ImgNet':
        root = '../../../disk2/zenghaien/imagenet/'
        # trainset = torchdata.ImageFolder(root+'train_100/', transform_train)
        # testset = torchdata.ImageFolder(root+'val_100/', transform_test)
        trainset = torchdata.ImageFolder(root + 'train_100_256/', transform_train)
        testset = torchdata.ImageFolder(root + 'val_100_256/', transform_test)
    elif dset=='SUN' or dset=='SUNhalf':
        root = '../../../disk2/zenghaien/SUN42/'
        trainset = torchdata.ImageFolder(root + 'train256/', transform_train)
        testset = torchdata.ImageFolder(root + 'test256/', transform_test)

    return trainset, testset

def get_sun_half():
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
    ])
    root = '../../../disk2/zenghaien/SUN42/'
    trainset = torchdata.ImageFolder(root + 'train256/', transform_train)
    testset = torchdata.ImageFolder(root + 'test256/', transform_test)
    return trainset,testset

def get_dataset_imgNet(root = '../../../disk2/zenghaien/imagenet/'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
    trainset = torchdata.ImageFolder(root + 'train_100/', transform_train)
    testset = torchdata.ImageFolder(root + 'val_100/', transform_test)

    return trainset, testset

# Make a new if statement for every new model variety you want to index
def get_model(model):

    from models import resnet, base, resnet_standard

    if model=='R32_C10':
        # rnet_checkpoint = 'cv/pretrained/R32_C10/pk_E_164_A_0.923.t7'
        # rnet_checkpoint = 'cv/pretrained/R32_C10/ckpt_E_4820_mAP_0.92652.t7'
        # rnet_checkpoint = 'cv/pretrained/R32_C10/ckpt_E_1860_mAP_0.87070.t7'
        rnet_checkpoint = 'cv/pretrained/R32_C10_jump/ckpt_E_4230_mAP_0.94279.t7'
        layer_config = [5, 5, 5]
        layer_config_small = [3,3,3]
        rnet = resnet.FlatResNet32_jump(base.BasicBlock, layer_config, num_classes=12)#64)
        target_rnet = resnet.FlatResNet32_jump(base.BasicBlock, layer_config, num_classes=32)#64)
        rnet_small = resnet.FlatResNet32(base.BasicBlock, layer_config_small, num_classes=32)
        agent = resnet.Policy32([1,1,1], num_blocks=15)
        agent_small = resnet.Policy32([1,1,1], num_blocks=9)
        agent_jump = resnet.PolicySize_2a([1, 1, 1], num_out=3)

        rnet_checkpoint = torch.load(rnet_checkpoint)
        rnet.load_state_dict(rnet_checkpoint)
        # target_rnet.load_state_dict(rnet_checkpoint)

    elif model=='R32_ImgNet':
        layer_config = [5, 5, 5]
        # layer_config_small = [3, 3, 3]
        rnet = resnet.FlatResNet32_jump(base.BasicBlock, layer_config, num_classes=64)  # 64)
        # target_rnet = resnet.FlatResNet32_jump(base.BasicBlock, layer_config, num_classes=12)  # 64)
        # rnet_small = resnet.FlatResNet32(base.BasicBlock, layer_config_small, num_classes=12)
        agent = resnet.Policy32([1, 1, 1], num_blocks=15)
        # agent_small = resnet.Policy32([1, 1, 1], num_blocks=9)
        agent_jump = resnet.PolicySize_2a([1, 1, 1], num_out=3)
        rnet_checkpoint = 'cv/pretrained/R32_ImgNet/ckpt_E_840_mAP_0.62668.t7'
        rnet_checkpoint = torch.load(rnet_checkpoint)
        rnet.load_state_dict(rnet_checkpoint)

    elif model=='R34_ImgNet':
        rnet = resnet_standard.resnet34(pretrained=True)
        agent = resnet.Policy32([1, 1, 1], num_blocks=16)

    elif model=='R50_ImgNet':
        rnet = resnet_standard.resnet50(pretrained=True)
        agent = resnet.Policy32([1, 1, 1], num_blocks=16)

    elif model=='R32_SUN' or model=='R32_SUNhalf':
        layer_config = [5, 5, 5]
        # layer_config_small = [3, 3, 3]
        rnet = resnet.FlatResNet32_jump(base.BasicBlock, layer_config, num_classes=32)  # 64)
        # target_rnet = resnet.FlatResNet32_jump(base.BasicBlock, layer_config, num_classes=12)  # 64)
        # rnet_small = resnet.FlatResNet32(base.BasicBlock, layer_config_small, num_classes=12)
        agent = resnet.Policy32([1, 1, 1], num_blocks=15)
        # agent_small = resnet.Policy32([1, 1, 1], num_blocks=9)
        agent_jump = resnet.PolicySize_2a([1, 1, 1], num_out=3)
        # rnet_checkpoint = 'cv/pretrained/R32_SUN/ckpt_E_60_mAP_0.56867.t7'
        # rnet_checkpoint = 'cv/pretrained/R32_SUN/ckpt_E_180_mAP_0.66008.t7'
        rnet_checkpoint = 'cv/pretrained/R32_SUN/ckpt_E_1150_mAP_0.79402.t7'
        rnet_checkpoint = torch.load(rnet_checkpoint)
        rnet.load_state_dict(rnet_checkpoint)

    elif model=='R34_SUN' or model=='R34_SUNhalf':
        rnet = resnet_standard.resnet34(pretrained=False)
        agent = resnet.Policy32([1, 1, 1], num_blocks=16)

    elif model=='R50_SUN' or model=='R50_SUNhalf':
        rnet = resnet_standard.resnet50(pretrained=False)
        agent = resnet.Policy32([1, 1, 1], num_blocks=16)

    elif model=='R110_C10':
        rnet_checkpoint = 'cv/pretrained/R110_C10/pk_E_130_A_0.932.t7'
        layer_config = [18, 18, 18]
        rnet = resnet.FlatResNet32_jump(base.BasicBlock, layer_config, num_classes=64)#12)
        target_rnet = resnet.FlatResNet32_jump(base.BasicBlock, layer_config, num_classes=64)#12)
        layer_config_small = [3, 3, 3]
        rnet_small = resnet.FlatResNet32(base.BasicBlock, layer_config_small, num_classes=64)
        agent = resnet.Policy32([1,1,1], num_blocks=54)
        agent_small = resnet.Policy32([1, 1, 1], num_blocks=9)
        agent_jump = resnet.PolicySize_2a([1, 1, 1], num_out=3)

    elif model=='R32_C100':
        rnet_checkpoint = 'cv/pretrained/R32_C100/pk_E_164_A_0.693.t7'
        layer_config = [5, 5, 5]
        rnet = resnet.FlatResNet32_jump(base.BasicBlock, layer_config, num_classes=100)
        target_rnet = resnet.FlatResNet32_jump(base.BasicBlock, layer_config, num_classes=100)
        layer_config_small = [3, 3, 3]
        rnet_small = resnet.FlatResNet32(base.BasicBlock, layer_config_small, num_classes=100)
        agent = resnet.Policy32([1,1,1], num_blocks=15)
        agent_small = resnet.Policy32([1, 1, 1], num_blocks=9)
        agent_jump = resnet.PolicySize_2a([1, 1, 1], num_out=3)

    elif model=='R110_C100':
        rnet_checkpoint = 'cv/pretrained/R110_C100/pk_E_160_A_0.723.t7'
        layer_config = [18, 18, 18]
        rnet = resnet.FlatResNet32_jump(base.BasicBlock, layer_config, num_classes=100)
        target_rnet = resnet.FlatResNet32_jump(base.BasicBlock, layer_config, num_classes=100)
        layer_config_small = [3, 3, 3]
        rnet_small = resnet.FlatResNet32(base.BasicBlock, layer_config_small, num_classes=100)
        agent = resnet.Policy32([1,1,1], num_blocks=54)
        agent_small = resnet.Policy32([1, 1, 1], num_blocks=9)
        agent_jump = resnet.PolicySize_2a([1, 1, 1], num_out=3)

    elif model=='R101_ImgNet':
        rnet_checkpoint = 'cv/pretrained/R101_ImgNet/ImageNet_R101_224_76.464'
        layer_config = [3,4,23,3]
        rnet = resnet.FlatResNet224(base.Bottleneck, layer_config, num_classes=1000)
        target_rnet = resnet.FlatResNet224(base.Bottleneck, layer_config, num_classes=1000)
        layer_config_small = [3, 3, 3]
        rnet_small = resnet.FlatResNet32(base.BasicBlock, layer_config_small, num_classes=1000)
        agent = resnet.Policy224([1,1,1,1], num_blocks=33)
        agent_small = resnet.Policy32([1, 1, 1], num_blocks=9)
        agent_jump = resnet.PolicySize_2a([1, 1, 1], num_out=3)

    # load pretrained weights into flat ResNet
    # rnet_checkpoint = torch.load(rnet_checkpoint)
    # rnet.load_state_dict(rnet_checkpoint)
    # target_rnet.load_state_dict(rnet_checkpoint)
    # load_weights_to_flatresnet(rnet_checkpoint, rnet)

    # return rnet, target_rnet, rnet_small, agent, agent_small, agent_jump
    # return rnet, target_rnet, agent
    return rnet, agent
    # return rnet, agent, agent_jump

def get_SUN32_model_small():
    from models import resnet, base, resnet_standard
    layer_config = [5, 5, 5]
    # layer_config_small = [3, 3, 3]
    rnet_small = resnet.FlatResNet32_jump(base.BasicBlock, layer_config, num_classes=32)  # 64)
    rnet_checkpoint = 'cv/pretrained/R32_SUNhalf/ckpt_E_3800_mAP_0.73907.t7'
    rnet_checkpoint = torch.load(rnet_checkpoint)
    rnet_small.load_state_dict(rnet_checkpoint)
    return rnet_small

def get_policyNet(size_num=2,branch_num=3):
    from models import policyNet
    policy_net = policyNet.PolicyNet(size_num=size_num,branch_num=branch_num)
    return policy_net

def timing(f):
    """print time used for function f"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        time_start = time.time()
        ret = f(*args, **kwargs)
        # print(f'total time = {time.time() - time_start:.4f}')
        print('total time = %.6f' % (time.time() - time_start))
        return ret

    return wrapper

@timing
def compute_mAP(trn_binary, tst_binary, trn_label, tst_label):
    """
    compute mAP by searching testset from trainset
    https://github.com/flyingpot/pytorch_deephash
    """
    for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        correct = (query_label == trn_label[query_result]).float()
        P = torch.cumsum(correct, dim=0) / Ns.float()
        AP.append(torch.sum(P * correct) / torch.sum(correct))
        # print(torch.sum(correct))
        # print(trn_binary.size(0))
    mAP = torch.mean(torch.Tensor(AP))
    return mAP

def compute_mAP_policy(trn_binary, tst_binary, trn_label, tst_label):
    """
    compute mAP by searching testset from trainset
    https://github.com/flyingpot/pytorch_deephash
    """
    for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

    # print(trn_binary.size())
    # print(tst_binary.size())
    # print(trn_label.size())
    # print(tst_label.size())
    # print(trn_binary)

    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    # print(tst_binary.size())
    # print(tst_label.size())
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        correct = (query_label == trn_label[query_result]).float()
        P = torch.cumsum(correct, dim=0) / Ns
        AP.append(torch.sum(P * correct) / torch.sum(correct))
        # print(torch.sum(correct))
        # print(trn_binary.size(0))
    mAP = torch.mean(torch.Tensor(AP))
    return mAP,torch.cuda.FloatTensor(AP)

def compute_mAP_topK(trn_binary, tst_binary, trn_label, tst_label, K):
    # for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

    AP = []
    # sort_idxs = []
    Ns = torch.arange(1, K + 1)
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        # sort_idxs.append(query_result.unsqueeze(0))
        correct = (query_label == trn_label[query_result[:K]]).float()
        P = torch.cumsum(correct, dim=0) / Ns.float()
        total_num = torch.sum(correct)
        den = float(K) if float(K)<total_num else total_num
        if den == 0:
            den = 0.1
        AP.append(torch.sum(P * correct) / den)  #torch.sum(correct)) ##
    mAP = torch.mean(torch.Tensor(AP))
    return mAP#,torch.cuda.FloatTensor(AP)#,torch.cat(sort_idxs)

def compute_mAP_train(trn_binary, all_binary, trn_label):
    """
    compute mAP in a batch
    """
    for x in trn_binary, trn_label: x.long()

    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    for i in range(trn_binary.size(0)):
        query_label, query_binary = trn_label[i], trn_binary[i]
        _, query_result = torch.sum((query_binary != all_binary).long(), dim=1).sort()
        correct = (query_label == trn_label[query_result]).float()
        P = torch.cumsum(correct, dim=0) / Ns
        AP.append(torch.sum(P * correct) / torch.sum(correct))
        # print(torch.sum(correct))
    mAP = torch.mean(torch.Tensor(AP))
    return mAP, torch.cuda.FloatTensor(AP)

@timing
def compute_mAP_MultiLabels(trn_binary, tst_binary, trn_label, tst_label):
    """
    compute mAP by searching testset from trainset
    https://github.com/flyingpot/pytorch_deephash
    """
    for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        # correct = (query_label == trn_label[query_result]).float()
        correct = ((trn_label[query_result]*query_label).sum(1)>0).float()
        P = torch.cumsum(correct, dim=0) / Ns
        AP.append(torch.sum(P * correct) / torch.sum(correct))
        # print(torch.sum(correct))
        # print(trn_binary.size(0))
    mAP = torch.mean(torch.Tensor(AP))
    return mAP

def compute_mAP_policy_MultiLabels(trn_binary, tst_binary, trn_label, tst_label):
    """
    compute mAP by searching testset from trainset
    https://github.com/flyingpot/pytorch_deephash
    """
    for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    # print(tst_binary.size())
    # print(tst_label.size())
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        # correct = (query_label == trn_label[query_result]).float()
        correct = ((trn_label[query_result] * query_label).sum(1) > 0).float()
        P = torch.cumsum(correct, dim=0) / Ns
        AP.append(torch.sum(P * correct) / torch.sum(correct))
        # print(torch.sum(correct))
        # print(trn_binary.size(0))
    mAP = torch.mean(torch.Tensor(AP))
    return mAP,torch.cuda.FloatTensor(AP)

def compute_mAP_train_MultiLabels(trn_binary, all_binary, trn_label):
    """
    compute mAP in a batch
    """
    for x in trn_binary, trn_label: x.long()

    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    for i in range(trn_binary.size(0)):
        query_label, query_binary = trn_label[i], trn_binary[i]
        _, query_result = torch.sum((query_binary != all_binary).long(), dim=1).sort()
        # correct = (query_label == trn_label[query_result]).float()
        correct = ((trn_label[query_result] * query_label).sum(1) > 0).float()
        P = torch.cumsum(correct, dim=0) / Ns
        AP.append(torch.sum(P * correct) / torch.sum(correct))
        # print(torch.sum(correct))
    mAP = torch.mean(torch.Tensor(AP))
    return mAP, torch.cuda.FloatTensor(AP)

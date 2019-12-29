# #加载resnet50的预训练模型
    # from torchvision.models import resnet50
    # rnet_checkpoint=torch.load('resnet18_wd2-1327-6654f50a.pth')
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
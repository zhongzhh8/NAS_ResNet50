import torch.nn as nn
import math


from .slimmable_ops import SwitchableBatchNorm2d
from .slimmable_ops import SlimmableConv2d, SlimmableLinear

depth=50
width_mult_list=[0.25, 0.5, 0.75, 1.0]
reset_parameters=False

class Block(nn.Module):
    def __init__(self, inp, outp, stride):
        super(Block, self).__init__()
        assert stride in [1, 2]

        midp = [i // 4 for i in outp]
        layers = [
            SlimmableConv2d(inp, midp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(midp),
            nn.ReLU(inplace=True),

            SlimmableConv2d(midp, midp, 3, stride, 1, bias=False),
            SwitchableBatchNorm2d(midp),
            nn.ReLU(inplace=True),

            SlimmableConv2d(midp, outp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(outp),
        ]
        self.body = nn.Sequential(*layers)

        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            self.shortcut = nn.Sequential(
                SlimmableConv2d(inp, outp, 1, stride=stride, bias=False),
                SwitchableBatchNorm2d(outp),
            )
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
            res += self.shortcut(x)
        res = self.post_relu(res)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224):
        super(Model, self).__init__()

        self.features = []
        # head
        assert input_size % 32 == 0

        # setting of inverted residual blocks
        self.block_setting_dict = {
            # : [stage1, stage2, stage3, stage4]
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        self.block_setting = self.block_setting_dict[depth]
        feats = [64, 128, 256, 512]
        channels = [int(64 * width_mult) for width_mult in width_mult_list]
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(
                    [3 for _ in range(len(channels))], channels, 7, 2, 3,
                    bias=False),
                SwitchableBatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
            )
        )

        # body
        for stage_id, n in enumerate(self.block_setting):
            outp = [
                int(feats[stage_id] * width_mult * 4)
                for width_mult in width_mult_list]
            for i in range(n):
                if i == 0 and stage_id != 0:
                    self.features.append(Block(channels, outp, 2))
                else:
                    self.features.append(Block(channels, outp, 1))
                channels = outp

        avg_pool_size = input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.outp = channels
        self.classifier = nn.Sequential(
            SlimmableLinear(
                self.outp,
                [num_classes for _ in range(len(self.outp))]
            )
        )
        if reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        outputs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name=='4' or name=='8' or name=='14':
                outputs.append(x)

        # x = self.features(x)
        last_dim = x.size()[1]
        x = x.view(-1, last_dim)
        outputs.append(x)
        # x = self.classifier(x) #我把这个注释掉。把最后的全连接层换成branchlast
        return outputs[0],outputs[1],outputs[2],outputs[3]

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class BranchLast(nn.Module):

    def __init__(self, inplanes=512, expansion=1,bits=48):
        super(BranchLast, self).__init__()
        in_channels = [512, 1024, 1536, 2048]  # 经过了stage4后得到的输入维度
        out_channels = [bits, bits, bits, bits]
        self.fc =nn.Sequential(SlimmableLinear(in_channels,out_channels),nn.Tanh())

    def forward(self, x):
        out = self.fc(x)
        return out


class BranchThird(nn.Module):

    def __init__(self, inplanes=256, expansion=1, bits=32):
        super(BranchThird, self).__init__()
        in_channels=[512,1024,1536,2048]#经过了stage4的第一个bottleneck后得到的输入维度
        out_channels=[bits,bits,bits,bits]
        #本来应该是自己写一个Bottleneck，但是我借用rnet的那个Bottleneck，就不需要自己写了。只需要接一个全连接层
        #同理branch1只需要写2个，branch2只需要写1个
        self.fc = nn.Sequential(SlimmableLinear(in_channels,out_channels),nn.Tanh())

    def forward(self, x):

        out = nn.AvgPool2d(x.size(2))(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class BranchSecond(nn.Module):

    def __init__(self, inplanes=256, expansion=1, bits=32):
        super(BranchSecond, self).__init__()
        in_channels=[256,512,768,1024] #经过了stage3的第一个bottleneck后得到的输入维度
        out_channels=[512,1024,1536,2048]
        self.bottleneck1 = Block(in_channels, out_channels, 2)
        in_channels=out_channels
        out_channels=[bits,bits,bits,bits]
        self.fc = nn.Sequential(SlimmableLinear(in_channels,out_channels),nn.Tanh())

    def forward(self, x):

        out = self.bottleneck1(x)

        out = nn.AvgPool2d(out.size(2))(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class BranchFirst(nn.Module):

    def __init__(self, inplanes=256, expansion=1, bits=32):
        super(BranchFirst, self).__init__()
        in_channels=[128,256,384,512]#经过了stage2的第一个bottleneck后得到的输入维度
        out_channels=[256,512,768,1024]
        self.bottleneck1=Block(in_channels, out_channels, 2)
        in_channels=out_channels
        out_channels=[512,1024,1536,2048]
        self.bottleneck2 = Block(in_channels, out_channels, 2)
        in_channels = out_channels
        out_channels = [bits,bits,bits,bits]
        self.fc = nn.Sequential(SlimmableLinear(in_channels,out_channels),nn.Tanh())

    def forward(self, x):

        out = self.bottleneck1(x)
        out = self.bottleneck2(out)

        out = nn.AvgPool2d(out.size(2))(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
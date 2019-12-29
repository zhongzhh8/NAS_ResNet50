import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = nn.AvgPool2d(x.size(2))(x)
        x = nn.AvgPool2d(x.size(2), stride=1)(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.fc(x)

        return x

    def forward_last(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.size())

        x = self.layer1(x)
        # print(x.size())
        x = self.layer2(x)
        # print(x.size())
        x = self.layer3(x)
        # print(x.size())
        x = self.layer4(x)
        # print(x.size())

        # print(x.size(2))
        x = nn.AvgPool2d(x.size(2),stride=1)(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())

        return  x

    def forward_all(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.size())

        out1 = self.layer1(x)
        # print(x.size())
        out2 = self.layer2(out1)
        # print(x.size())
        out3 = self.layer3(out2)
        # print(x.size())
        out4 = self.layer4(out3)
        # print(x.size())

        # print(x.size(2))
        out4 = nn.AvgPool2d(out4.size(2),stride=1)(out4)
        # print(x.size())
        out4 = out4.view(out4.size(0), -1)
        # print(x.size())

        return  out1,out2,out3,out4

    def forward_jump1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        return x

    def forward_jump2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        return x

    def forward_jump3(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class ResNet_C10(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_C10, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        nn.AvgPool2d(x.size(2))(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward_last(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = nn.AvgPool2d(x.size(2),stride=1)(x)
        x = x.view(x.size(0), -1)

        return  x

    def forward_all(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # print(x.size())

        out1 = self.layer1(x)
        # print(x.size())
        out2 = self.layer2(out1)
        # print(x.size())
        out3 = self.layer3(out2)
        # print(x.size())
        out4 = self.layer4(out3)
        # print(x.size())

        # print(x.size(2))
        out4 = nn.AvgPool2d(out4.size(2),stride=1)(out4)
        # print(x.size())
        out4 = out4.view(out4.size(0), -1)
        # print(x.size())

        return  out1,out2,out3,out4

    def forward_jump1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)

        return x

    def forward_jump2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)

        return x

    def forward_jump3(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

class BranchFirst(nn.Module):

    def __init__(self, inplanes=64, expansion=1, bits=32):
        super(BranchFirst, self).__init__()
        self.conv1 = nn.Conv2d(inplanes * expansion, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, bits)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = nn.AvgPool2d(out.size(2))(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def forward_dropout(self, x):

        out = self.conv1(x)
        out = nn.Dropout(0.5)(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = nn.Dropout(0.5)(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = nn.AvgPool2d(out.size(2))(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class LCFirst(nn.Module):

    def __init__(self, inplanes=64, expansion=1, bits=32):
        super(LCFirst, self).__init__()
        self.conv1 = nn.Conv2d(inplanes * expansion, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256, bits)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = nn.AvgPool2d(out.size(2))(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class BranchSecond(nn.Module):

    def __init__(self, inplanes=128, expansion=1, bits=32):
        super(BranchSecond, self).__init__()
        self.conv1 = nn.Conv2d(inplanes*expansion, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512, bits)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = nn.AvgPool2d(out.size(2))(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def forward_dropout(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = nn.Dropout(0.5)(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = nn.AvgPool2d(out.size(2))(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class LCSecond(nn.Module):

    def __init__(self, inplanes=128, expansion=1, bits=32):
        super(LCSecond, self).__init__()
        self.conv1 = nn.Conv2d(inplanes*expansion, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(256, bits)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = nn.AvgPool2d(out.size(2))(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class BranchThird(nn.Module):

    def __init__(self, inplanes=256, expansion=1, bits=32):
        super(BranchThird, self).__init__()
        self.conv1 = nn.Conv2d(inplanes*expansion, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512, bits)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = nn.AvgPool2d(out.size(2))(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def forward_dropout(self, x):

        out = self.conv1(x)
        out = nn.Dropout(0.5)(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = nn.AvgPool2d(out.size(2))(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class LCThird(nn.Module):

    def __init__(self, inplanes=256, expansion=1, bits=32):
        super(LCThird, self).__init__()
        self.fc = nn.Linear(256, bits)

    def forward(self, x):

        out = nn.AvgPool2d(x.size(2))(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class BranchLast(nn.Module):

    def __init__(self, inplanes=512, expansion=1,bits=32):
        super(BranchLast, self).__init__()
        self.fc = nn.Linear(inplanes*expansion, bits)

    def forward(self, x):
        out = self.fc(x)

        return out

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet10(pretrained=False, **kwargs):

    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        pass
    return model


def resnet34_c10(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_C10(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pass
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet18_c10(pretrained=False, **kwargs):

    model = ResNet_C10(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pass
    return model

def resnet10_c10(pretrained=False, **kwargs):

    model = ResNet_C10(BasicBlock, [1, 1, 1, 1], **kwargs)
    if pretrained:
        pass
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

class ResNetFirst(nn.Module):

    def __init__(self, block, block_num=3):
        self.inplanes = 64
        super(ResNetFirst, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        return x

class ResNetSecond(nn.Module):

    def __init__(self, block, block_num=4):
        self.inplanes = 64
        super(ResNetSecond, self).__init__()

        self.layer2 = self._make_layer(block, 128, block_num, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer2(x)

        return x


class ResNetThird(nn.Module):

    def __init__(self, block, block_num=6):
        self.inplanes = 128
        super(ResNetThird, self).__init__()

        self.layer3 = self._make_layer(block, 256, block_num, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer3(x)

        return x


class ResNetLast(nn.Module):

    def __init__(self, block, block_num=3):
        self.inplanes = 256
        super(ResNetLast, self).__init__()

        self.layer4 = self._make_layer(block, 512, block_num, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer4(x)
        x = nn.AvgPool2d(x.size(2))(x)
        x = x.view(x.size(0), -1)
        return x

class TestFirstBranch(nn.Module):
    def __init__(self):
        super(TestFirstBranch, self).__init__()
        self.featureExtractor = ResNetFirst(BasicBlock,block_num=3)
        self.hashGenerator = BranchFirst()
    def forward(self,x):
        x = self.featureExtractor(x)
        x = self.hashGenerator(x)
        return x

class TestThirdBranch(nn.Module):
    def __init__(self):
        super(TestThirdBranch, self).__init__()
        self.featureExtractor = nn.Sequential(
            ResNetFirst(BasicBlock, block_num=3),
            ResNetSecond(BasicBlock, block_num=4),
            ResNetThird(BasicBlock,block_num=6)
        )
        self.hashGenerator = BranchFirst()
        self.fc = nn.Linear(256, 32)
    def forward(self,x):
        x = self.featureExtractor.forward(x)
        x = nn.AvgPool2d(x.size(2))(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print(x)
        return x

class TestLastBranch(nn.Module):
    def __init__(self):
        super(TestLastBranch, self).__init__()
        self.featureExtractor = nn.Sequential(
            ResNetFirst(BasicBlock, block_num=3),
            ResNetSecond(BasicBlock, block_num=4),
            ResNetThird(BasicBlock,block_num=6),
            ResNetLast(BasicBlock, block_num=3)
        )
        self.hashGenerator = BranchLast()
        # self.fc = nn.Linear(512, 32)
    def forward(self,x):
        x = self.featureExtractor.forward(x)
        # print(x)
        x = self.hashGenerator.forward(x)
        # print(x)
        return x

class TestFirstBranch2(nn.Module):
    def __init__(self, block=BasicBlock, block_num=3, bits=32):
        super(TestFirstBranch2, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_num)
        self.conv1_b = nn.Conv2d(self.inplanes * block.expansion, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_b = nn.BatchNorm2d(128)
        self.conv2_b = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2_b = nn.BatchNorm2d(256)
        self.conv3_b = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3_b = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, bits)
        # self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        out = self.conv1_b(x)
        out = self.bn1_b(out)
        out = self.relu(out)

        out = self.conv2_b(out)
        out = self.bn2_b(out)
        out = self.relu(out)

        out = self.conv3_b(out)
        out = self.bn3_b(out)
        out = self.relu(out)

        out = nn.AvgPool2d(out.size(2))(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def resnet34_separated(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    resnetFirstPart = ResNetFirst(BasicBlock, 3)
    resnetSecondPart = ResNetSecond(BasicBlock,4)
    resnetThirdPart = ResNetThird(BasicBlock,6)
    resnetLastPart = ResNetLast(BasicBlock,3)
    return resnetFirstPart,resnetSecondPart,resnetThirdPart,resnetLastPart


def resnet50_separated(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    resnetFirstPart = ResNetFirst(Bottleneck, 3)
    resnetSecondPart = ResNetSecond(Bottleneck, 4)
    resnetThirdPart = ResNetThird(Bottleneck, 6)
    resnetLastPart = ResNetLast(Bottleneck, 3)
    return resnetFirstPart, resnetSecondPart, resnetThirdPart, resnetLastPart
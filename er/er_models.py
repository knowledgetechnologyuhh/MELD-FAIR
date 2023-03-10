import torch
from torch import nn
from torch.nn import functional as F

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class audioEncoder(nn.Module):
    def __init__(self, layers, num_filters, **kwargs):
        super(audioEncoder, self).__init__()
        block = SEBasicBlock
        self.inplanes   = num_filters[0]

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))
        out_dim = num_filters[3] * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        x = torch.mean(x, dim=2, keepdim=True)
        x = x.view((x.size()[0], x.size()[1], -1))
        x = x.transpose(1, 2)

        return x


class ResNetLayer(nn.Module):
    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """
    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1,1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return

    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch


class ResNet(nn.Module):
    """
    An 18-layer ResNet architecture.
    """
    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(1,1))
        
        return


    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch


class GlobalLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + 1e-8, 0.5) + self.beta
        return gLN_y


class attentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(attentionLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src2 = self.self_attn(tar, src, src, attn_mask=None, key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1) # T, B, C -> B, T, C
        return src


class ERModel_Audio(nn.Module):
    def __init__(self, num_classes):
        super(EmoRecogModel, self).__init__()
        self.audioEncoder  = audioEncoder(layers = [3, 4, 6, 3],  num_filters = [64, 128, 256, 512])
        self.self_att = attentionLayer(d_model = 512, nhead = 8)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, a):
        a = a.unsqueeze(1).transpose(2, 3)
        a = self.audioEncoder(a)
        a = a.transpose(1, 2)
        a = torch.max(a, -1)[0].unsqueeze(1)
        a = self.self_att(a, a)[:, 0, :]
        a = self.fc(a)
        return a


class ERModel_Video(nn.Module):
    def __init__(self, num_classes, channel_mean, channel_std):
        super(EmoRecogModel, self).__init__()
        self.channel_mean = channel_mean
        self.channel_std = channel_std
        # Visual Temporal Encoder
        self.visualFrontend  = visualFrontend() # Visual Frontend
        self.visualTCN       = visualTCN()      # Visual Temporal Network TCN
        self.visualConv1D    = visualConv1D()   # Visual Temporal Network Conv1d
        self.self_att = attentionLayer(d_model = 512, nhead = 8)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        B, T, C, W, H = x.shape
        x = x.view(B * T, 1, C, W, H)
        x = x.transpose(2, 3).transpose(3, 4)
        x = (x - self.channel_mean) / self.channel_std
        x = x.transpose(4, 3).transpose(3, 2)
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)
        x = x.transpose(1, 2)
        x = self.visualTCN(x)
        x = torch.max(x, -1)[0].unsqueeze(1)
        x = self.self_att(x, x)[:, 0, :]
        x = self.fc(x)
        return x


class ERModel_BiModal(nn.Module):
    def __init__(self, num_classes, channel_mean, channel_std):
        super(EmoRecogModel, self).__init__()
        self.channel_mean = channel_mean
        self.channel_std = channel_std
        self.audioEncoder  = audioEncoder(layers = [3, 4, 6, 3],  num_filters = [64, 128, 256, 512])
        # Visual Temporal Encoder
        self.visualFrontend  = visualFrontend() # Visual Frontend 
        self.visualTCN       = visualTCN()      # Visual Temporal Network TCN
        self.visualConv1D    = visualConv1D()   # Visual Temporal Network Conv1d
        self.self_att = attentionLayer(d_model = 1024, nhead = 8)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, v, a):
        # Video
        B, T, C, W, H = v.shape
        v = v.view(B * T, 1, C, W, H)
        v = v.transpose(2, 3).transpose(3, 4)
        v = (v - self.channel_mean) / self.channel_std
        v = v.transpose(4, 3).transpose(3, 2)
        v = self.visualFrontend(v)
        v = v.view(B, T, 512)
        v = v.transpose(1, 2)
        v = self.visualTCN(v)
        v = torch.max(v, -1)[0]
        
        # Audio
        a = a.unsqueeze(1).transpose(2, 3)
        a = self.audioEncoder(a)
        a = a.transpose(1, 2)
        a = torch.max(a, -1)[0]
        
        av = torch.cat((a, v), -1).unsqueeze(1)
        av = self.self_att(av, av)[:, 0, :]
        av = self.fc(av)
        return av

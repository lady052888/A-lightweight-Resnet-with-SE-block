import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# remember modify the number of channel, Ensure this matches the number of input signal in 'segmenter' function in dataset.py

# Squeeze-and-Excitation Block (Channel Attention)
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=2):
        """
        Channel Attention (Squeeze-and-Excitation Block)
        Args:
        - in_channels: Number of input channels
        - reduction: Channel reduction ratio
        """
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _ = x.size()
        # Squeeze: Global average pooling -> [batch_size, channels]
        y = self.global_avg_pool(x).view(batch_size, channels)
        # Excitation: Fully connected layers -> [batch_size, channels]
        y = self.fc(y).view(batch_size, channels, 1)
        # Reweight: Apply weights -> [batch_size, channels, length]
        return x * y


# BasicBlock with optional SE Block
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p=0.5, use_se=False, reduction=16):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.dropout1 = nn.Dropout(p)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.dropout2 = nn.Dropout(p)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

        # Add SE Block (optional)
        self.use_se = use_se
        if self.use_se:
            self.se_block = SEBlock(planes, reduction)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        if self.use_se:
            out = self.se_block(out)  # Apply SE Block
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, p=0.5, use_se=False, reduction=16):
        super(ResNet1D, self).__init__()
        self.in_planes = 32 

        self.conv1 = nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)  # Input channels set to 1，2 or 3(means PPG as input/ PPG+first as input /PPG+first+second as input), output channels to 32
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p)

        # ResNet layers with optional SE Block
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1, p=p, use_se=use_se, reduction=reduction)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2, p=p, use_se=use_se, reduction=reduction)
       
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)  

    def _make_layer(self, block, planes, num_blocks, stride, p, use_se, reduction):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, p, use_se, reduction))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
       
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    

# ResNet18 with optional Channel Attention 
def ResNet18(num_classes=2, p=0.2, use_se=False, reduction=16):
    return ResNet1D(BasicBlock1D, [2, 2], num_classes=num_classes, p=p, use_se=use_se, reduction=reduction)
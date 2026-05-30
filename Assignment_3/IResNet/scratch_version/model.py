"""
IResNet (Improved Residual Network) — From-Scratch Implementation
=================================================================
Paper: "Improved Residual Networks for Image and Video Recognition" (Duta et al., 2020)

This module implements the three key innovations of iResNet over standard ResNet:

1. RESTRUCTURED BN/ReLU PLACEMENT
   - Blocks use start_block / end_block / exclude_bn0 flags to avoid
     the "ReLU after addition" pattern that clips negative gradients
     through skip connections.
   
   Block positions within a stage:
     First  (start_block=True):  Conv→BN→ReLU→Conv→BN_last→ +identity
     Second (exclude_bn0=True):  ReLU→Conv→BN→ReLU→Conv→ +identity
     Middle (exclude_bn0=False): BN→ReLU→Conv→BN→ReLU→Conv→ +identity
     Last   (end_block=True):    BN→ReLU→Conv→BN→ReLU→Conv→ +identity→BN→ReLU

2. MAXPOOL-BASED PROJECTION SHORTCUT
   - When downsampling spatially: MaxPool(3×3, stride) → Conv1×1 → BN
   - Instead of ResNet's strided Conv1×1 → BN (which discards 75% of spatial info)

3. NO SEPARATE MAXPOOL IN STEM
   - ResNet: conv7×7(s=2) → BN → ReLU → MaxPool(s=2) → layer1(s=1)
   - iResNet: conv7×7(s=2) → BN → ReLU → layer1(s=2)
   - For CIFAR-10 (32×32 inputs): conv3×3(s=1) → BN → ReLU → layer1(s=1)
"""

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Basic residual block (2 × 3×3 conv) with iResNet's restructured BN/ReLU.
    
    The block's forward path changes based on its position within a stage:
    - start_block: first block  → no pre-BN/ReLU, has trailing BN on residual
    - end_block:   last block   → has trailing BN + ReLU after addition
    - exclude_bn0: second block → skips bn0, applies ReLU directly
    - default:     middle block → full pre-activation (BN→ReLU before first conv)
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None,
                 start_block=False, end_block=False, exclude_bn0=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # bn0 is the pre-activation BN applied before the first conv
        # It is NOT created for the start_block (no pre-activation needed)
        # or when exclude_bn0 is True (second block — just ReLU, no BN)
        if not start_block and not exclude_bn0:
            self.bn0 = norm_layer(inplanes)

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        # Trailing BN: only for start_block and end_block
        if start_block:
            self.bn2 = norm_layer(planes)
        if end_block:
            self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def forward(self, x):
        identity = x

        # --- Pre-activation logic (before first conv) ---
        if self.start_block:
            # First block: no pre-activation, go straight to conv
            out = self.conv1(x)
        elif self.exclude_bn0:
            # Second block: apply ReLU only (the un-ReLU'd output from start_block's BN)
            out = self.relu(x)
            out = self.conv1(out)
        else:
            # Middle blocks: full pre-activation with BN → ReLU
            out = self.bn0(x)
            out = self.relu(out)
            out = self.conv1(out)

        # --- Standard mid-block path ---
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        # --- Trailing BN for start_block (before addition) ---
        if self.start_block:
            out = self.bn2(out)

        # --- Shortcut / downsample ---
        if self.downsample is not None:
            identity = self.downsample(x)

        # --- Residual addition ---
        out += identity

        # --- Trailing BN + ReLU for end_block (after addition) ---
        if self.end_block:
            out = self.bn2(out)
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block (1×1 → 3×3 → 1×1) with iResNet's restructured BN/ReLU.
    Same positional logic as BasicBlock but with 3 convolutions.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None,
                 start_block=False, end_block=False, exclude_bn0=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if not start_block and not exclude_bn0:
            self.bn0 = norm_layer(inplanes)

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)

        if start_block:
            self.bn3 = norm_layer(planes * self.expansion)
        if end_block:
            self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def forward(self, x):
        identity = x

        if self.start_block:
            out = self.conv1(x)
        elif self.exclude_bn0:
            out = self.relu(x)
            out = self.conv1(out)
        else:
            out = self.bn0(x)
            out = self.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.start_block:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.end_block:
            out = self.bn3(out)
            out = self.relu(out)

        return out


class IResNet(nn.Module):
    """
    Improved Residual Network (iResNet).
    
    For CIFAR-10 (32×32), uses a 3×3 stride-1 stem without MaxPool.
    For ImageNet (224×224), uses a 7×7 stride-2 stem without MaxPool
    (layer1 handles downsampling via stride=2 instead).
    """

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 norm_layer=None, dropout_prob0=0.0, is_cifar=True):
        super(IResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64

        if is_cifar:
            # CIFAR-10: 3×3 conv, stride 1, no MaxPool
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            layer1_stride = 1  # No downsampling at layer1 for CIFAR
        else:
            # ImageNet: 7×7 conv, stride 2, NO MaxPool (unlike ResNet)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            layer1_stride = 2  # Layer1 does the downsampling that MaxPool would do

        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        # Build the four stages
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=layer1_stride, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if dropout_prob0 > 0.0:
            self.dp = nn.Dropout(dropout_prob0, inplace=True)
        else:
            self.dp = None

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-init last BN for identity-like initial behavior
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        """
        Build one stage of the network.
        
        KEY DIFFERENCE from ResNet:
        - Downsample uses MaxPool + Conv1×1 instead of strided Conv1×1
        - Blocks use start_block/end_block/exclude_bn0 for BN/ReLU restructuring
        """
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        downsample = None

        # ---- Build downsample path (iResNet's MaxPool-based projection) ----
        if stride != 1 and self.inplanes != planes * block.expansion:
            # Need both spatial downsampling AND channel projection
            downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            # Channel projection only (no spatial change)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        elif stride != 1:
            # Spatial downsampling only (no channel change)
            downsample = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        # ---- Build the blocks with positional flags ----
        layers = []

        # First block: start_block=True
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer,
                            start_block=True))
        self.inplanes = planes * block.expansion

        # Middle blocks
        exclude_bn0 = True  # Second block skips bn0
        for _ in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                                exclude_bn0=exclude_bn0))
            exclude_bn0 = False  # Subsequent middle blocks have full pre-activation

        # Last block: end_block=True
        layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                            end_block=True, exclude_bn0=exclude_bn0))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.dp is not None:
            x = self.dp(x)

        x = self.fc(x)
        return x


# ============================================================================
# Factory functions for CIFAR-10
# ============================================================================

def iresnet18_cifar(num_classes=10, **kwargs):
    """iResNet-18 for CIFAR-10 (BasicBlock, [2,2,2,2])"""
    return IResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, is_cifar=True, **kwargs)


def iresnet34_cifar(num_classes=10, **kwargs):
    """iResNet-34 for CIFAR-10 (BasicBlock, [3,4,6,3])"""
    return IResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, is_cifar=True, **kwargs)


def iresnet50_cifar(num_classes=10, **kwargs):
    """iResNet-50 for CIFAR-10 (Bottleneck, [3,4,6,3])"""
    return IResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, is_cifar=True, **kwargs)

import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn.functional as F
import torch
from dataset import SyntheticDS, synth_transforms
from model import RegressionHead
backbone = mobilenet_v3_small(MobileNet_V3_Small_Weights.DEFAULT)

class DepthwiseSepConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw = nn.Sequential( # depthwise convolution, just spatial mixing
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish()
        )
        self.pw = nn.Sequential( # pointwise convolutions, just channel mixing
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish()
        )
    def forward(self, x):
        return self.pw(self.dw(x))
    
class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.decode_conv = DepthwiseSepConv(in_channels, out_channels)
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.decode_conv(x)
        return x
    

class UNet_Decoder(nn.Module):
    def __init__(self, backbone, out_channels):
        super().__init__()
        self.backbone = backbone
        self.blocks = []
        self.strided = []
        self.segments = []

        last_skip = 0
        layers = list(backbone.features.children())

        entry = layers[0][0] # adjust for single-channel grayscale entry. MobileNetV3 was originally for RGB input
        layers[0][0] = nn.Conv2d(
            in_channels = 1,
            out_channels = entry.out_channels,
            kernel_size = entry.kernel_size,
            stride = entry.stride,
            padding = entry.padding,
            bias = entry.bias is not None
        )

        for i, layer in enumerate(layers):
            for module in layer.modules():
                logged = False
                if isinstance(module, nn.Conv2d) and not logged:
                    if module.stride != (1, 1):
                        logged = True
                        self.strided.append(layer.out_channels) # for each strided layer of backbone, record output channel count
                        
                        self.segments.append(nn.Sequential(*layers[last_skip:i+1])) # also segment the backbone by skip point, so we can call them sequentially in forward
                        last_skip = i + 1

        self.strided = self.strided[::-1]
        for (o1, o2) in (self.strided[x:x+2] for x in range(len(self.strided)-1)): # for array of [96, 40, 24, 16, 16], make block for [96, 40], [40, 24], so on
            self.blocks.append(DecodeBlock(o1 + o2, o2))

        self.heatmap_head = nn.Conv2d(in_channels=self.strided[-1], out_channels=out_channels, kernel_size=1)
        self.diameter_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # normalized to [0,1] range
        )
        self.gaze_head = RegressionHead(in_features=96, out_dim=3)
        self.gaze_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        
        skips = []
        # start with a normal pass through the backbone, segment by segment, and saving at skip points
        for segment in self.segments:
            x = segment(x)
            skips.append(x)

        gaze = self.gaze_head(self.gaze_pool(skips.pop()).flatten(1)) # last segment ends at bottleneck level, so pop it and feed to gaze

        skips = skips[::-1] # reverse order of skips, deep -> shallow now.
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i])
        
        heatmaps = F.hardsigmoid(self.heatmap_head(x))
        diameter = self.diameter_head(x)

        return gaze, heatmaps, diameter


device = torch.device('cpu')
a = SyntheticDS(transforms=synth_transforms)
b = UNet_Decoder(backbone, out_channels=17).to(device, dtype=torch.float32)


g,h,d = b(a[0][0].to(device, dtype=torch.float32).unsqueeze(0))

print(h.shape)
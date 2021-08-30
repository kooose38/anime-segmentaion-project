import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import List, Tuple, Any, Dict

########################################################################################
class conv2normRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int, stride: int, padding: int, dilation: int,  bias: bool):
        super(conv2normRelu, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding=padding, bias=bias, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class conv2norm(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int, stride: int, padding: int=1, dilation: int=1, bias: bool=False):
        super(conv2norm, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding=padding, bias=bias, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.norm(self.conv(x))
########################################################################################


class FeatureMap_convolution(nn.Module):
    def __init__(self):
        '''
        入力された画像データの特長変換をする
        (b, 3, 475, 475) -> (b, 128, 119, 119)
        '''
        super(FeatureMap_convolution, self).__init__()
        self.layer1 = conv2normRelu(3, 64, 3, 2, 1, 1,False)
        self.layer2 = conv2normRelu(64, 64, 3, 1, 1, 1, False)
        self.layer3 = conv2normRelu(64, 128, 3, 1, 1, 1, False)
        self.layer4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out 
    
    
class ResidentBlockPSP(nn.Sequential):
    def __init__(self, n_block: int, in_c: int, mid_c: int, out_c: int, kernel_size: int, stride: int=1, dilation: int=1):
        '''ResNet50による畳み込みとスキップコネクションを挟む'''
        super(ResidentBlockPSP, self).__init__()

        self.add_module("block1", bottleNeckPSP(in_c, mid_c, out_c, stride, dilation))
        for i in range(n_block-1):
            self.add_module(f"block{str(i+2)}", bottleNeckIdentifyPSP(out_c, mid_c, stride, dilation))

            
class bottleNeckPSP(nn.Module):
    def __init__(self, in_c, mid_c, out_c, stride=1, dilation=1):
        super(bottleNeckPSP, self).__init__()
        self.layer1 = conv2normRelu(in_c, mid_c, 1, 1, 0, 1, False)
        self.layer2 = conv2normRelu(mid_c, mid_c, 3, stride, dilation, dilation, False)
        self.layer3 = conv2normRelu(mid_c, out_c, 1, 1, 0, 1, False)

        self.skip = conv2normRelu(in_c, out_c, 1, stride, 0, 1, False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        '''
        (b, in_c, w, h) -> (b, out_c, w/stride, h/stride)
        '''
        out = self.layer3(self.layer2(self.layer1(x)))
        residual = self.skip(x) # skip connection
        return self.relu(out+residual)

    
class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_c: int, mid_c: int, stride: int, dilation: int):
        super(bottleNeckIdentifyPSP, self).__init__()
        self.layer1 = conv2normRelu(in_c, mid_c, 1, 1, 0, 1, False)
        self.layer2 = conv2normRelu(mid_c, mid_c, 3, 1, dilation, dilation, False)
        self.layer3 = conv2normRelu(mid_c, in_c, 1, 1, 0, 1, False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        '''
        (b, c, w, h) -> (b, c, w, h)
        '''
        out = self.layer3(self.layer2(self.layer1(x)))
        return self.relu(out+x) # skip connection 
    
    
########################################################################################
class Adapt2conv2normRelu(nn.Module):
    def __init__(self, pool_size, in_c, out_c, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(Adapt2conv2normRelu, self).__init__()
        self.adapt = nn.AdaptiveAvgPool2d(output_size=pool_size) # w, h -> out, out
        self.conv = conv2normRelu(in_c, out_c, kernel_size, stride, padding, dilation, bias)

    def forward(self, x):
        out = self.adapt(x)
        out = self.conv(out)
        return out 
########################################################################################
    

class PyramidPooling(nn.Module):
    def __init__(self, in_c, pool_size: List[int]):
        '''入力された画像に対してそれぞれ独立して畳み込みをする'''
        super(PyramidPooling, self).__init__()

        output_size = int(in_c/len(pool_size))

        self.layer1 = Adapt2conv2normRelu(pool_size[0], in_c, output_size)
        self.layer2 = Adapt2conv2normRelu(pool_size[1], in_c, output_size)
        self.layer3 = Adapt2conv2normRelu(pool_size[2], in_c, output_size)
        self.layer4 = Adapt2conv2normRelu(pool_size[3], in_c, output_size)

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        out1 = self.layer1(x) # -> (b, 512, 6, 6)
        out1 = F.interpolate(out1, size=(h, w), mode="bilinear", align_corners=True) # (b, 512, 60, 60)
        out2 = self.layer2(x) # -> (b, 512, 3, 3)
        out2 = F.interpolate(out2, size=(h, w), mode="bilinear", align_corners=True) # (b, 512, 60, 60)
        out3 = self.layer3(x) # -> (b, 512, 2, 2)
        out3 = F.interpolate(out3, size=(h, w), mode="bilinear", align_corners=True) # (b, 512, 60, 60)
        out4 = self.layer4(x) # -> (b, 512, 1, 1)
        out4 = F.interpolate(out4, size=(h, w), mode="bilinear", align_corners=True) # (b, 512, 60, 60)

        return torch.cat((x, out1, out2, out3, out4), dim=1) # (b, 4096, 60, 60)

    
class DecoderPSPFeature(nn.Module):
    def __init__(self, tag_size: int, img_size: int, in_c: int, out_c: int):
        '''元の画像サイズにアンプーリングする層'''
        super(DecoderPSPFeature, self).__init__()
        self.conv = conv2normRelu(in_c, out_c, 3, 1, 1, 1, False)
        self.drop = nn.Dropout(0.1)
        self.classification = nn.Conv2d(out_c, tag_size, 1, 1, 0)
    
    def forward(self, x):
        out = self.classification(self.drop(self.conv(x)))
        out = F.interpolate(out, size=(self.img_size, self.img_size), mode="bilinear", align_corners=True)
        return out 
    
    
class PSPNet(nn.Module):
    def __init__(self, tag_size: int):
        super(PSPNet, self).__init__()

        block = [3, 4, 6, 3]
        img_size = 475

        self.feature = FeatureMap_convolution()
        self.res_1 = ResidentBlockPSP(block[0], 128, 64, 256, 1, 1)
        self.res_2 = ResidentBlockPSP(block[1], 256, 128, 512, 2, 1)
        self.res_3 = ResidentBlockPSP(block[2], 512, 256, 1024, 1, 2)
        self.res_4 = ResidentBlockPSP(block[3], 1024, 512, 2048, 1, 4)

        self.pyramid = PyramidPooling(2048, [6, 3, 2, 1])

        self.aux = DecoderPSPFeature(tag_size, img_size, 1024, 256)
        self.decoder = DecoderPSPFeature(tag_size, img_size, 4096, 512)

    def forward(self, x):
        out = self.feature(x) # -> (b, 128, 119, 119)
        out = self.res_1(out) # -> (b, 256, 119, 119)
        out = self.res_2(out) # -> (b, 512, 60, 60)
        out = self.res_3(out) # -> (b, 1024, 60, 60)

        aux_out = out 
        out_ = self.aux(aux_out) # -> (b, tag_size, 475, 475)
        out = self.res_4(out) # -> (b, 2048, 60, 60)

        out = self.pyramid(out) # -> (b, 4096, 60, 60)
        out = self.decoder(out) # -> (b, tag_size, 475, 475)
        return out, out_
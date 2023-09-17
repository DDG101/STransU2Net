# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils




@manager.MODELS.add_component
class STransformer(nn.Layer):


    def __init__(self, num_classes=2, in_ch=3, pretrained=None):
        super(STransformer, self).__init__()

        #swintransformer
        self.swin1 = layers.SwinTW(in_channels=64, out_channels=64, input_resolution=(512,512), num_heads=8, window_size=8, downsample=False)
        self.swin2 = layers.SwinTW(in_channels=128, out_channels=128, input_resolution=(256,256), num_heads=8, window_size=8, downsample=False)
        self.swin3 = layers.SwinTW(in_channels=256, out_channels=256, input_resolution=(128,128), num_heads=8, window_size=8, downsample=False)
        self.swin4 = layers.SwinTW(in_channels=512, out_channels=512, input_resolution=(64,64), num_heads=8, window_size=8, downsample=False)
        self.swin5 = layers.SwinTW(in_channels=512, out_channels=512, input_resolution=(32,32), num_heads=8, window_size=8, downsample=False)
        self.swin6 = layers.SwinTW(in_channels=512, out_channels=512, input_resolution=(16,16), num_heads=8, window_size=8, downsample=False)

        self.stage1 = RSU7(in_ch, 32, 64)
        self.cbame1 = CBAM(64)
        self.pool12 = DownPool(64)

        self.stage2 = RSU6(64, 32, 128)
        self.cbame2 = CBAM(128)
        self.pool23 = DownPool(128)

        self.stage3 = RSU5(128, 64, 256)
        self.cbame3 = CBAM(256)
        self.pool34 = DownPool(256)

        self.stage4 = RSU4(256, 128, 512)
        self.cbame4 = CBAM(512)        
        self.pool45 = DownPool(512)

        self.stage5 = RSU4F(512, 256, 512)
        self.cbame5 = CBAM(512)              
        self.pool56 = DownPool(512)

        self.stage6 = RSU4F(512, 256, 512)
        self.cbame6 = CBAM(512)   

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.cbam5d = CBAM(512)  
        self.stage4d = RSU4(1024, 128, 256)
        self.cbam4d = CBAM(256)      
        self.stage3d = RSU5(512, 64, 128)
        self.cbam3d = CBAM(128)  
        self.stage2d = RSU6(256, 32, 64)
        self.cbam2d = CBAM(64)  
        self.stage1d = RSU7(128, 16, 64)
        self.cbam1d = CBAM(64)  

        self.side1 = nn.Conv2D(64, num_classes, 3, padding=1)
        self.side2 = nn.Conv2D(64, num_classes, 3, padding=1)
        self.side3 = nn.Conv2D(128, num_classes, 3, padding=1)
        self.side4 = nn.Conv2D(256, num_classes, 3, padding=1)
        self.side5 = nn.Conv2D(512, num_classes, 3, padding=1)
        self.side6 = nn.Conv2D(512, num_classes, 3, padding=1)

        self.outconv = nn.Conv2D(6 * num_classes, num_classes, 1)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx1 = self.cbame1(hx1)

        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx2 = self.cbame2(hx2)
        hx2_swin = self.swin2(hx2)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx3 = self.cbame3(hx3)
        hx3_swin = self.swin3(hx3)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx4 = self.cbame4(hx4)
        hx4_swin = self.swin4(hx4)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx5 = self.cbame5(hx5)
        hx5_swin = self.swin5(hx5)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6 = self.cbame6(hx6)
        # hx6 = self.swin6(hx6)

        hx6up = _upsample_like(hx6, hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(paddle.concat((hx6up, hx5_swin), 1))
        hx5d = self.cbam5d(hx5d)
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(paddle.concat((hx5dup, hx4_swin), 1))
        hx4d = self.cbam4d(hx4d)
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(paddle.concat((hx4dup, hx3_swin), 1))
        hx3d = self.cbam3d(hx3d)
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(paddle.concat((hx3dup, hx2_swin), 1))
        hx2d = self.cbam2d(hx2d)
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(paddle.concat((hx2dup, hx1), 1))
        hx1d = self.cbam1d(hx1d)

        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(paddle.concat((d1, d2, d3, d4, d5, d6), 1))


        return [d0, d1, d2, d3, d4, d5, d6]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class DownPool(nn.Layer):
    def __init__(self,in_channels,ratio=4):
        super().__init__()
        self.max_pool =  nn.MaxPool2D(kernel_size = 3, stride = 2,padding = 1)
        self.down_fanpool =  nn.Sequential(
            nn.Conv2D(in_channels, in_channels // ratio, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2D(in_channels // ratio),
            nn.Conv2D(in_channels // ratio, in_channels // ratio, kernel_size=3, stride=2,padding=1,groups = in_channels//ratio),
            nn.BatchNorm2D(in_channels // ratio),
            nn.Conv2D(in_channels // ratio, in_channels, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2D(in_channels),
            nn.ReLU()
        )
    def forward(self,x):
        x1 = self.max_pool(x)
        x2 = self.down_fanpool(x)
        out = x1 + x2
        return out



class REBNCONV(nn.Layer):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2D(
            in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2D(out_ch)
        self.relu_s1 = nn.ReLU()

    def forward(self, x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):

    src = F.upsample(src, size=paddle.shape(tar)[2:], mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Layer):  #UNet07DRES(nn.Layer):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(paddle.concat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(paddle.concat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(paddle.concat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(paddle.concat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(paddle.concat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(paddle.concat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Layer):  #UNet06DRES(nn.Layer):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(paddle.concat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(paddle.concat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(paddle.concat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(paddle.concat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(paddle.concat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Layer):  #UNet05DRES(nn.Layer):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(paddle.concat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(paddle.concat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(paddle.concat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(paddle.concat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Layer):  #UNet04DRES(nn.Layer):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2D(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(paddle.concat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(paddle.concat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(paddle.concat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Layer):  #UNet04FRES(nn.Layer):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(paddle.concat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(paddle.concat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(paddle.concat((hx2d, hx1), 1))

        return hx1d + hxin

class ChannelAttention(nn.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)

        self.fc1 = nn.Conv2D(in_planes, in_planes // ratio, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2D(in_planes // ratio, in_planes,kernel_size=1, stride=1)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2D(2, 1, kernel_size, stride=1, padding=3)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Layer):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = paddle.multiply(x ,self.ca(x))
        result = paddle.multiply(out , self.sa(out))
        return result

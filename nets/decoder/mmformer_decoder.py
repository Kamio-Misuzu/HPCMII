import torch
import torch.nn as nn

basic_dims = 8

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class general_conv3d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d_prenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)
        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class fusion_prenorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(fusion_prenorm, self).__init__()
        self.fusion_layer = nn.Sequential(
            general_conv3d_prenorm(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
            general_conv3d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
            general_conv3d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1)
        )

    def forward(self, x):
        return self.fusion_layer(x)


class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_fuse, self).__init__()

        self.d4_c1 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims*16, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims*4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims*16, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims*2, num_cls=num_cls)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims*1, num_cls=num_cls)

    def forward(self, x1, x2, x3, x4, x5):

        de_x5 = self.RFM5(x5)                       # (B, 128, 10, 12, 8)
        pred4 = self.softmax(self.seg_d4(de_x5))    # (B, num_cls, 10, 12, 8)
        de_x5 = self.d4_c1(self.up2(de_x5))         # (B, 64, 20, 24, 16)


        de_x4 = self.RFM4(x4)                      # (B, 64, 20, 24, 16)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)   # (B, 128, 20, 24, 16)
        de_x4 = self.d4_out(self.d4_c2(de_x4))     # (B, 64, 20, 24, 16)
        pred3 = self.softmax(self.seg_d3(de_x4))   # (B, num_cls, 20, 24, 16)
        de_x4 = self.d3_c1(self.up2(de_x4))        # (B, 32, 40, 48, 32)


        de_x3 = self.RFM3(x3)                      # (B, 32, 40, 48, 32)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)   # (B, 64, 40, 48, 32)
        de_x3 = self.d3_out(self.d3_c2(de_x3))     # (B, 32, 40, 48, 32)
        pred2 = self.softmax(self.seg_d2(de_x3))   # (B, num_cls, 40, 48, 32)
        de_x3 = self.d2_c1(self.up2(de_x3))        # (B, 16, 80, 96, 64)


        de_x2 = self.RFM2(x2)                      # (B, 16, 80, 96, 64)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)   # (B, 32, 80, 96, 64)
        de_x2 = self.d2_out(self.d2_c2(de_x2))     # (B, 16, 80, 96, 64)
        pred1 = self.softmax(self.seg_d1(de_x2))   # (B, num_cls, 80, 96, 64)
        de_x2 = self.d1_c1(self.up2(de_x2))        # (B, 8, 160, 192, 128)

        de_x1 = self.RFM1(x1)                      # (B, 8, 160, 192, 128)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)   # (B, 16, 160, 192, 128)
        de_x1 = self.d1_out(self.d1_c2(de_x1))     # (B, 8, 160, 192, 128)

        logits = self.seg_layer(de_x1)             # (B, num_cls, 160, 192, 128)
        pred = self.softmax(logits)                # (B, num_cls, 160, 192, 128)

        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))


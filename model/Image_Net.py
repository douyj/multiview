import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


class Stage1UNet(nn.Module):
    def __init__(self, img_channel, width, middle_blk_num, enc_blk_nums, dec_blk_nums):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[MDPRBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.lgag = LGAG(chan, chan, chan // 2)

        self.middle_blks = nn.Sequential(
            *[MDPRBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[MDPRBlock(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        _, _, H, W = inp.shape
        x_in = self.check_image_size(inp)

        x = self.intro(x_in)

        encs = []
        decs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        x_v = torch.flip(x, dims=[3])
        x = self.lgag(x_v, x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            decs.append(x)

        x = self.ending(x)
        x = x + x_in

        return x[:, :, :H, :W], encs, decs[::-1]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class SSFNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, bias, num_cab):
        super().__init__()

        self.orb1 = nn.Sequential(
            *[MDPRBlock(n_feat + scale_orsnetfeats) for _ in range(num_cab)]
        )
        self.orb2 = nn.Sequential(
            *[MDPRBlock(n_feat + scale_orsnetfeats) for _ in range(num_cab)]
        )
        self.orb3 = nn.Sequential(
            *[MDPRBlock(n_feat + scale_orsnetfeats) for _ in range(num_cab)]
        )

        self.up_enc1 = UpSample(n_feat * 2)
        self.up_dec1 = UpSample(n_feat * 2)

        self.up_enc2 = nn.Sequential(UpSample(n_feat * 4), UpSample(n_feat * 2))
        self.up_dec2 = nn.Sequential(UpSample(n_feat * 4), UpSample(n_feat * 2))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


class SAM(nn.Module):
    """
    改成单通道 CT 版本：
    原来是 n_feat -> 3 -> n_feat
    现在改成 n_feat -> 1 -> n_feat
    """
    def __init__(self, n_feat, kernel_size, bias):
        super().__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 1, kernel_size, bias=bias)
        self.conv3 = conv(1, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        attn = torch.sigmoid(self.conv3(img))
        x1 = x1 * attn
        x1 = x1 + x
        return x1, img


class SCABlock(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, act):
        super().__init__()
        self.body = nn.Sequential(
            conv(n_feat, n_feat, kernel_size, bias),
            act,
            conv(n_feat, n_feat, kernel_size, bias),
        )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=n_feat,
                out_channels=n_feat,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

    def forward(self, x):
        res = self.body(x)
        res = res * self.sca(res)
        res = res + x
        return res

# 主模型
class ImageRestorer(nn.Module):
    def __init__(self, in_c=1, out_c=1, stage1_width=64, stage2_width=64, num_cab=6):
        super().__init__()

        # 1. 两个浅层特征提取
        self.shallow_feat1 = nn.Sequential(
            conv(in_c, stage1_width, 3),
            SCABlock(stage1_width, 3, False, nn.ReLU(inplace=True)),
        )
        self.shallow_feat2 = nn.Sequential(
            conv(in_c, stage2_width, 3),
            SCABlock(stage2_width, 3, False, nn.ReLU(inplace=True)),
        )

        # 2. 第一阶段：UNet 主干网络- 粗恢复
        self.stage1 = Stage1UNet(
            img_channel=in_c,
            width=stage1_width,
            middle_blk_num=4,
            enc_blk_nums=[2, 2, 2, 2],
            dec_blk_nums=[2, 2, 2, 2],
        )

        # 第二阶段精修网络 SSFNet
        self.orsnet = SSFNet(stage1_width, stage2_width, False, num_cab)

        # SAM注意力模块
        self.sam12 = SAM(stage1_width, kernel_size=1, bias=False)

        # 特征融合 + 输出层
        self.concat12 = nn.Conv2d(
            stage2_width + stage1_width,
            stage2_width + stage1_width,
            3,
            padding=1,
        )
        self.tail = nn.Conv2d(stage2_width + stage1_width, out_c, 3, padding=1)     #输出1通道CT图

    def forward(self, x):
        """
        x:
            1通道: [B,1,H,W] = I_fbp
            2通道: [B,2,H,W] = [I_fbp, U_sino_img]

        注意:
            主干网络 stage1 / shallow_feat2 可以吃完整 x；
            但 SAM 和最终残差连接只能用第 1 通道 I_fbp。
        """

        # 输入CT图
        x_base = x[:, 0:1, :, :]

        # Stage1 主干吃完整输入 x
        # 如果 in_c=2，这里就是 [I_fbp, U_sino_img]
        stage1_img, encs_feature, decs_feature = self.stage1(x)     #输出：[stage1_img粗恢复CT, encs_feature编码器多层特征, decs_feature解码器多层特征]

        # SAM 里的 img 分支是 1通道，所以这里只能传 x_base
        stage1_samfeats, _ = self.sam12(decs_feature[0], x_base)

        # Stage2 浅层特征也吃完整输入 x
        stage2_feat = self.shallow_feat2(x)

         # 融合两阶段特征，拼接：浅层特征 + 注意力精炼特征
        stage2_cat = self.concat12(
            torch.cat([stage2_feat, stage1_samfeats], dim=1)
        )

        # 第二阶段精修网络 SSFNet，融合 U-Net 全部多尺度信息，去残留伪影
        x2_cat = self.orsnet(stage2_cat, encs_feature, decs_feature)

         # 输出最终图像
        final_img = self.tail(x2_cat)

        # 残差连接：加上原始FBP图，保证结构不变
        final_out = final_img + x_base

       # 只输出第一通道作为结果
        stage1_out = stage1_img[:, 0:1, :, :]

        return final_out, stage1_out


class MDPRBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.ca = CAB(dw_channel // 2, dw_channel // 2)
        self.sa = SpatialAttention()
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.ca(x)
        x = self.sa(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        ratio = min(ratio, self.in_channels)
        self.reduced_channels = max(self.in_channels // ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = activation
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_pool, max_pool], dim=1)
        attention_map = self.sigmoid(self.conv(spatial_features))
        return x * attention_map


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, -1, 1, 1) * y + bias.view(1, -1, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, -1, 1, 1)
        mean_g = g.mean(1, keepdim=True)
        mean_gy = (g * y).mean(1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum((0, 2, 3)), grad_output.sum((0, 2, 3)), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation=nn.ReLU(inplace=True)):
        super().__init__()

        if kernel_size == 1:
            groups = 1

        gn_groups = 8 if F_int >= 8 else 1

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.GroupNorm(gn_groups, F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.GroupNorm(gn_groups, F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid(),
        )
        self.activation = activation

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        return x * psi
# class LGAG(nn.Module):
#     def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation=nn.ReLU(inplace=True)):
#         super().__init__()

#         if kernel_size == 1:
#             groups = 1

#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups, bias=True),
#             nn.BatchNorm2d(F_int),
#         )
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups, bias=True),
#             nn.BatchNorm2d(F_int),
#         )
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid(),
#         )
#         self.activation = activation

#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.activation(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.up(x)
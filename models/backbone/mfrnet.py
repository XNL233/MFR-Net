import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from einops import rearrange


# UPro Generator
class UFProGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UFProGenerator, self).__init__()
        assert num_downs == 8

        unet_block8 = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, norm_layer=norm_layer, innermost=True)
        unet_block7 = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block6 = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block5 = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block4 = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer)
        unet_block3 = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer)
        unet_block2 = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer)
        unet_block1 = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, outermost=True, norm_layer=norm_layer)

        self.down1, self.up1 = unet_block1.down, unet_block1.up
        self.down2, self.up2 = unet_block2.down, unet_block2.up
        self.down3, self.up3 = unet_block3.down, unet_block3.up
        self.down4, self.up4 = unet_block4.down, unet_block4.up
        self.down5, self.up5 = unet_block5.down, unet_block5.up
        self.down6, self.up6 = unet_block6.down, unet_block6.up
        self.down7, self.up7 = unet_block7.down, unet_block7.up
        self.down8, self.up8 = unet_block8.down, unet_block8.up

        self.dfd = DynamicFrequencyDecomposition(in_c=3, basic_dim=ngf)

        # prompt module
        self.prompt3 = PromptModule(basic_dim=ngf, dim=int(ngf*2**3), input_resolution=8)
        self.prompt2 = PromptModule(basic_dim=ngf, dim=int(ngf*2**2), input_resolution=32)
        self.prompt1 = PromptModule(basic_dim=ngf, dim=ngf, input_resolution=128)

    def forward(self, input, encode_only=False):
        if input.shape[1] == 6:
            img = input[:, :3]
            x = input
        elif input.shape[1] == 3:
            img = input
            x = input
        else:
            raise ValueError(f'Expected 3 or 6 input channels, got {input.shape[1]}')

        # subsampled
        d1 = self.down1(x)  # [1, ngf, 128, 128]
        d2 = self.down2(d1)  # [1, ngf*2, 64, 64]
        d3 = self.down3(d2)  # [1, ngf*4, 32, 32]
        d4 = self.down4(d3)  # [1, ngf*8, 16, 16]
        # d5 = self.down5(d4)  # [1, ngf*8, 8, 8]

        # get deep features
        # *** Important: use features before InstanceNorm ***
        d5_pre = self.down5[0](d4)  # LeakyReLU(0.2)
        d5_pre = self.down5[1](d5_pre)  # Conv2d(512→512)
        if encode_only:  # just use the encoder
            return d5_pre  # [1, ngf*8, 8, 8]

        d5 = self.down5[2](d5_pre)  # InstanceNorm2d

        d6 = self.down6(d5)  # [1, ngf*8, 4, 4]
        d7 = self.down7(d6)  # [1, ngf*8, 2, 2]
        d8 = self.down8(d7)  # [1, ngf*8, 1, 1]

        low_part, out_high = self.dfd(img)  # [1, ngf, 256, 256]  [1, ngf, 256, 256]

        # upsampling
        u8 = self.up8(d8)  # [1, ngf*8, 2, 2]
        u7 = self.up7(torch.cat([u8, d7], 1))  # [1, ngf*8, 4, 4]
        u6 = self.up6(torch.cat([u7, d6], 1))  # [1, ngf*8, 8, 8]
        u6_p = self.prompt3(low_part, out_high, u6) + u6  # [1, ngf*8, 8, 8]

        u5 = self.up5(torch.cat([u6_p, d5], 1))  # [1, ngf*8, 16, 16]
        u4 = self.up4(torch.cat([u5, d4], 1))  # [1, ngf*4, 32, 32]
        u4_p = self.prompt2(low_part, out_high, u4) + u4  # [1, ngf*4, 32, 32]

        u3 = self.up3(torch.cat([u4_p, d3], 1))  # [1, ngf*2, 64, 64]
        u2 = self.up2(torch.cat([u3, d2], 1))  # [1, ngf, 128, 128]
        u2_p = self.prompt1(low_part, out_high, u2) + u2  # [1, ngf, 128, 128]

        u1 = self.up1(torch.cat([u2_p, d1], 1))  # [1, 3, 256, 256]

        return u1, d5_pre  # return image and deep feature
        # return u1


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                up = up + [nn.Dropout(0.5)]

        self.up = nn.Sequential(*up)
        self.down = nn.Sequential(*down)


# **************************** Dual Prompt Block *******************************
# window operation
def window_partition(x, win_size):
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True, isQuery=True):
        super().__init__()
        self.isQuery = isQuery
        inner_dim = dim_head * heads
        self.heads = heads
        if self.isQuery:
            self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        else:
            self.to_kv = nn.Linear(dim, 2 * inner_dim, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        if self.isQuery:
            q = self.to_q(x)
            q = q.reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
            q = q[0]
            return q
        else:
            C = self.inner_dim
            kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1,4).contiguous()
            k, v = kv[0], kv[1]
            return k, v


# Dynamic Frequency Division Module (DFDM)
class DynamicFrequencyDecomposition(nn.Module):
    def __init__(self, in_c, basic_dim=32):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_c, basic_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.dyna_channel = DynamicGroupedLowPassFilter(inchannels=basic_dim)

    def forward(self, F_low):
        _, c_basic, h_ori, w_ori = F_low.shape
        F_low = self.patch_embed(F_low)
        low_part, out_high = self.dyna_channel(F_low)
        return low_part, out_high

class DynamicGroupedLowPassFilter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(DynamicGroupedLowPassFilter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.conv_gate = nn.Conv2d(group * kernel_size ** 2, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.act_gate = nn.Sigmoid()

        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)

        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        identity_input = x
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = low_filter * self.act_gate(self.conv_gate(low_filter))
        low_filter = self.bn(low_filter)

        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group, self.kernel_size ** 2, h * w)
        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)
        low_filter = self.act(low_filter)

        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_high = identity_input - low_part
        return low_part, out_high


# Frequency Prompt Guidance Module (FPGM)
class PromptModule(nn.Module):
    def __init__(self, basic_dim=32, dim=32, input_resolution=128):
        super().__init__()
        h = input_resolution
        w = input_resolution // 2 + 1
        self.simple = nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1)

        self.FPG_h = FrequencyPromptGenerator(basic_dim, h, w, is_high_frequency=True)
        self.FPG_l = FrequencyPromptGenerator(basic_dim, h, w, is_high_frequency=False)

        self.CA_h = HighFrequencyPromptFusion(dim, basic_dim, win_size=8, num_heads=2, bias=False)
        self.CA_l = LowFrequencyPromptFusion(dim, basic_dim, num_heads=2, bias=False)

        self.FPI = FrequencyPromptInteraction(dim)

    def forward(self, low_part, out_high, x):
        b, c, h, w = x.shape
        y_h = self.FPG_h(out_high, h, w)
        y_l = self.FPG_l(low_part, h, w)
        y_h = self.CA_h(x, y_h)
        y_l = self.CA_l(x, y_l)

        # x = self.simple(torch.cat([y_h, y_l], dim=1))  # not interact
        x = self.FPI(y_l, y_h)

        return x

# Frequency Prompt Generation (FPG)
class FrequencyPromptGenerator(nn.Module):
    def __init__(self, dim=3, h=128, w=65, is_high_frequency=True):
        super().__init__()
        self.hf = is_high_frequency
        k_size = 3
        if is_high_frequency:
            w = (w - 1) * 2
            self.w = w
            self.h = h
            self.prompt_h = nn.Parameter(torch.randn(1, dim, h, w, dtype=torch.float32) * 0.02)
            self.body = nn.Sequential(nn.Conv2d(dim, dim, (1, k_size), padding=(0, k_size // 2), groups=dim),
                                      nn.Conv2d(dim, dim, (k_size, 1), padding=(k_size // 2, 0), groups=dim),
                                      nn.GELU())
        else:
            self.prompt_l = nn.Parameter(torch.randn(1, dim, h, w, 2, dtype=torch.float32) * 0.02)
            self.body = nn.Sequential(nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, stride=1),
                                      nn.GELU(), )

    def forward(self, ffm, H, W):
        if self.hf:  # high-frequency branch
            ffm = F.interpolate(ffm, size=(H, W), mode='bilinear')
            y_att = self.body(ffm)

            y_f = y_att * ffm
            weight = torch.tanh(self.prompt_h)
            y = y_f * (1.0 + weight)
        else:  # low-frequency branch
            ffm = F.interpolate(ffm, size=(H, W), mode='bicubic')
            y = torch.fft.rfft2(ffm.float(), norm='ortho')
            y_imag = y.imag
            y_real = y.real
            y_f = torch.cat([y_real, y_imag], dim=1)
            weight = torch.complex(torch.tanh(self.prompt_l[..., 0]),
                                   torch.tanh(self.prompt_l[..., 1]))
            y_att = self.body(y_f)
            y_f = y_f * y_att
            y_real, y_imag = torch.chunk(y_f, 2, dim=1)
            y = torch.complex(y_real, y_imag)
            y = y * (1.0 + weight)
            y = torch.fft.irfft2(y, s=(H, W), norm='ortho')
        return y

# Frequency Prompt Interaction (FPI)
class FrequencyPromptInteraction(nn.Module):
    def __init__(self, dim):
        super(FrequencyPromptInteraction, self).__init__()
        self.H_L_Gate = H_LGate()
        self.L_H_Gate = L_HGate(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)
    def forward(self, low, high):
        H_L_weight = self.H_L_Gate(high)
        L_H_weight = self.L_H_Gate(low)
        high = high * L_H_weight
        low = low * H_L_weight

        out = low + high
        out = self.conv(out)
        return out

class LowFrequencyPromptFusion(nn.Module):
    def __init__(self, dim, dim_bak, num_heads, bias=False):
        super(LowFrequencyPromptFusion, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.ap_kv = nn.AdaptiveAvgPool2d(1)
        self.kv = nn.Conv2d(dim_bak, dim * 2, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, feature, prompt_feature):
        b, c1, h, w = feature.shape
        _, c2, _, _ = prompt_feature.shape

        query = self.q(feature).reshape(b, h * w, self.num_heads, c1 // self.num_heads).permute(0, 2, 1, 3).contiguous()

        prompt_feature = self.ap_kv(prompt_feature)  # .reshape(b, c2, -1).permute(0, 2, 1)
        key_value = self.kv(prompt_feature).reshape(b, 2 * c1, -1).permute(0, 2, 1).contiguous()
        key_value = key_value.reshape(b, -1, 2, self.num_heads, c1 // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        key, value = key_value[0], key_value[1]

        attn = (query @ key.transpose(-2, -1).contiguous()) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ value)
        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class HighFrequencyPromptFusion(nn.Module):
    def __init__(self, dim, dim_bak, win_size, num_heads, qkv_bias=True, qk_scale=None, bias=False):
        super(HighFrequencyPromptFusion, self).__init__()
        self.num_heads = num_heads
        self.win_size = win_size  # Wh, Ww
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_q = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias, isQuery=True)
        self.to_kv = LinearProjection(dim_bak, num_heads, dim // num_heads, bias=qkv_bias, isQuery=False)

        self.kv_dwconv = nn.Conv2d(dim_bak, dim_bak, kernel_size=3, stride=1, padding=1, groups=dim_bak, bias=bias)
        self.softmax = nn.Softmax(dim=-1)
        self.project_out = nn.Linear(dim, dim)

    def forward(self, query_feature, key_value_feature):
        b, c, h, w = query_feature.shape
        _, c_2, _, _ = key_value_feature.shape

        key_value_feature = self.kv_dwconv(key_value_feature)

        # partition windows
        query_feature = rearrange(query_feature, ' b c1 h w -> b h w c1 ', h=h, w=w)
        query_feature_windows = window_partition(query_feature, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        query_feature_windows = query_feature_windows.view(-1, self.win_size * self.win_size,c)  # nW*B, win_size*win_size, C

        key_value_feature = rearrange(key_value_feature, ' b c2 h w -> b h w c2 ', h=h, w=w)
        key_value_feature_windows = window_partition(key_value_feature,self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        key_value_feature_windows = key_value_feature_windows.view(-1, self.win_size * self.win_size,c_2)  # nW*B, win_size*win_size, C

        B_, N, C = query_feature_windows.shape

        query = self.to_q(query_feature_windows)
        query = query * self.scale

        key, value = self.to_kv(key_value_feature_windows)
        attn = (query @ key.transpose(-2, -1).contiguous())
        attn = attn.softmax(dim=-1)
        out = (attn @ value).transpose(1, 2).contiguous().reshape(B_, N, C)
        out = self.project_out(out)

        attn_windows = out.view(-1, self.win_size, self.win_size, C)
        attn_windows = window_reverse(attn_windows, self.win_size, h, w)  # B H' W' C
        return rearrange(attn_windows, 'b h w c -> b c h w', h=h, w=w)

# High-frequency Compensation (HC)
class H_LGate(nn.Module):
    def __init__(self):
        super(H_LGate, self).__init__()
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        max = torch.max(x,1,keepdim=True)[0]
        mean = torch.mean(x,1,keepdim=True)
        scale = torch.cat((max, mean), dim=1)
        scale = self.spatial(scale)  # 7x7 conv
        scale = F.sigmoid(scale)  # sigmoid
        return scale  # (batch_size, 1, H, W)

# Low-frequency Compensation (LC)
class L_HGate(nn.Module):
    def __init__(self, dim):
        super(L_HGate, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.max = nn.AdaptiveMaxPool2d((1,1))
        self.mlp = nn.Sequential(nn.Conv2d(dim, dim//16, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(dim//16, dim, 1, bias=False))
    def forward(self, x):
        avg = self.mlp(self.avg(x))
        max = self.mlp(self.max(x))
        scale = avg + max
        scale = F.sigmoid(scale)  # sigmoid
        return scale  # (batch_size, channels, 1, 1)



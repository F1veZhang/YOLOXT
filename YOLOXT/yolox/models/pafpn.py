#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .darknet import CSPDarknet
from .network_blocks import *
from .deform_conv_v2 import *
from .swin import *



device = torch.device('cuda:0')

class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features   # 记录dark345的输出
        self.in_channels = in_channels   # 记录dark345的通道
        Conv = DWConv if depthwise else BaseConv   # 分离卷积和普通卷积

        self.DCN = DeformConv2d(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1,bias=False, modulation=True)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")     # 最临近插值进行上采样
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.swin1 = nn.ModuleList([
            SwinTransformerBlock(
                dim=512,
                num_heads=2,
                window_size=7,
                shift_size=0 if (i % 2 == 0) else 3,
                mlp_ratio=4,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=nn.LayerNorm)
            for i in range(2)])
        self.swin2 = nn.ModuleList([
            SwinTransformerBlock(
                dim=256,
                num_heads=4,
                window_size=7,
                shift_size=0 if (i % 2 == 0) else 3,
                mlp_ratio=4,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=nn.LayerNorm)
            for i in range(2)])
        self.swin3 = nn.ModuleList([
            SwinTransformerBlock(
                dim=256,
                num_heads=8,
                window_size=7,
                shift_size=0 if (i % 2 == 0) else 3,
                mlp_ratio=4,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=nn.LayerNorm)
            for i in range(2)])

        self.patch_embed1 = PatchEmbed(
            patch_size=1, in_c=512, embed_dim=512,
            norm_layer=nn.LayerNorm)
        self.patch_embed2 = PatchEmbed(
            patch_size=1, in_c=256, embed_dim=256,
            norm_layer=nn.LayerNorm)
        self.patch_embed3 = PatchEmbed(
            patch_size=1, in_c=256, embed_dim=256,
            norm_layer=nn.LayerNorm)
        self.ffim1 = Fusion_1()
        self.ffim2 = Fusion_2()

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / 7)) * 7
        Wp = int(np.ceil(W / 7)) * 7
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=None)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        w_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, 7)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, 7 * 7)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        #input = input.to(device)
        out_features = self.backbone(input)
        
        features = [out_features[f] for f in self.in_features]
        
        [x2, x1, x0] = features
        # 这里加入
        #print(f"Device tensor is stored on: {x0.device}")
        # x0 = x0.to(device)
        b, c, H, W = x0.shape
        # print(f"sssDevice tensor is stored on: {x0.device}")
        x0, H, W = self.patch_embed1(x0)  # tuple
        #print(f"sssDevice tensor is stored on: {x0.device}")
        
        ################################4_17 modify: adjust attn_mask to x0  ###################################
        attn_mask = self.create_mask(x=x0, H=H, W=W).to(x0.device)
        # print(f"Device tensor is stored on: {attn_mask.device}")
        #attn_mask = attn_mask.to(device)
        
        for blk in self.swin1:
            blk.H, blk.W = H, W
             #x0=torch.tensor([x0.detach().numpy() for i in x0])
            x0 = blk(x0, attn_mask)
            # print(f"sssDevice tensor is stored on: {x0.device}")
        x0 = x0.permute(0, 2, 1).view(b, c, H, W)
        # print(f"aaaDevice tensor is stored on: {x0.device}")
        

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32 卷积
        #f_out0 = self.upsample(fpn_out0)  # 512/16 进行上采样的操作
        #f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16 concat操作
        f_out0 = self.ffim1(fpn_out0, x1)
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        
        b, c, H, W = f_out0.shape
        f_out0, H, W = self.patch_embed2(f_out0)  # tuple
        attn_mask = self.create_mask(x=f_out0, H=H, W=W).to(f_out0.device)
        for blk in self.swin2:
            blk.H, blk.W = H, W
            # x0=torch.tensor([x0.detach().numpy() for i in x0])
            f_out0 = blk(f_out0, attn_mask)
        f_out0 = f_out0.permute(0, 2, 1).view(b, c, H, W)
        
        
        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        #f_out1 = self.upsample(fpn_out1)  # 256/8
        #f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        f_out1 = self.ffim2(fpn_out1, fpn_out0, x2)
		
        
        b, c, H, W = f_out1.shape
        f_out1, H, W = self.patch_embed3(f_out1)  # tuple
        attn_mask = self.create_mask(x=f_out1, H=H, W=W).to(f_out1.device)
        for blk in self.swin3:
            blk.H, blk.W = H, W
            # x0=torch.tensor([x0.detach().numpy() for i in x0])
            f_out1 = blk(f_out1, attn_mask)
        f_out1 = f_out1.permute(0, 2, 1).view(b, c, H, W)
        

        pan_out2 = self.C3_p3(f_out1)  # 512->256/8
        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs  # 返回三个输出pan_out2（80*80*128）, pan_out1（40*40*256）, pan_out0（20*20*512）

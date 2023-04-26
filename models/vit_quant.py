# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import collections.abc
import functools
import math
import os
import re
import warnings
from collections import OrderedDict
from functools import partial
from itertools import repeat

import torch
import torch.nn.functional as F
from torch import nn

from .layers_quant import DropPath, HybridEmbed, Mlp, PatchEmbed, trunc_normal_
from .ptq import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear
from .utils import load_weights_from_npz



__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'vit_base_patch16_224', 'vit_large_patch16_224'
]


#revised by chengquan

def hook_actin(module, input, result, filename):
    # 获取中间层的权重和激活值
    if not torch.is_tensor(module.enquantdata):
        weight = module.enquantdata
        weight_txt = weight.reshape(-1)
        #activation = output.data.cpu().numpy()
        # 将权重和激活值保存到二进制文件中
        with open('./export/' + filename + '_float.bin', 'wb') as f:
            f.write(weight.tobytes())

        with open('./export/' + filename + '_float.txt', 'w') as f:
            for item in weight_txt:
                f.write("%s\n" % item)

    if not torch.is_tensor(module.dequantdata):
        weight = module.dequantdata
        weight_txt = weight.reshape(-1)
        #activation = output.data.cpu().numpy()
        # 将权重和激活值保存到二进制文件中
        with open('./export/' + filename + '_int8.bin', 'wb') as f:
            f.write(weight.tobytes())

        with open('./export/' + filename + '_int8.txt', 'w') as f:
            for item in weight_txt:
                f.write("%s\n" % item)

    if not torch.is_tensor(module.scale):
        weight = module.scale
        weight_txt = weight.reshape(-1)
        with open('./export/' + filename + '_scale_cfg.txt', 'w') as f:
            for item in weight_txt:
                f.write("%s\n" % item)

    if not torch.is_tensor(module.zero_point):
        weight = module.zero_point
        weight_txt = weight.reshape(-1)
        with open('./export/' + filename + '_zeropoint_cfg.txt', 'w') as f:
            for item in weight_txt:
                f.write("%s\n" % item)  


def hook_linear(module, input, result, filename):
    # 获取中间层的权重和激活值
    if not torch.is_tensor(module.enquantdata):
        weight = module.enquantdata
        weight_txt = weight.reshape(-1)
        #activation = output.data.cpu().numpy()
        # 将权重和激活值保存到二进制文件中
        with open('./export/' + filename + '_wfloat.bin', 'wb') as f:
            f.write(weight.tobytes())

        with open('./export/' + filename + '_wfloat.txt', 'w') as f:
            for item in weight_txt:
                f.write("%s\n" % item)

    if not torch.is_tensor(module.dequantdata):
        weight = module.dequantdata
        weight_txt = weight.reshape(-1)
        #activation = output.data.cpu().numpy()
        # 将权重和激活值保存到二进制文件中
        with open('./export/' + filename + '_wint.bin', 'wb') as f:
            f.write(weight.tobytes())

        with open('./export/' + filename + '_wint.txt', 'w') as f:
            for item in weight_txt:
                f.write("%s\n" % item)

    if not torch.is_tensor(module.scale):
        weight = module.scale
        weight_txt = weight.reshape(-1)
        with open('./export/' + filename + '_scale_cfg.txt', 'w') as f:
            for item in weight_txt:
                f.write("%s\n" % item)

    if not torch.is_tensor(module.zero_point):
        weight = module.zero_point
        weight_txt = weight.reshape(-1)
        with open('./export/' + filename + '_zeropoint_cfg.txt', 'w') as f:
            for item in weight_txt:
                f.write("%s\n" % item)  

    if not torch.is_tensor(module.bias_data):
        weight = module.bias_data
        weight_txt = weight.reshape(-1)
        with open('./export/' + filename + '_bias.txt', 'w') as f:
            for item in weight_txt:
                f.write("%s\n" % item)  

    if not torch.is_tensor(module.dataout):
        weight = module.dataout
        weight_txt = weight.reshape(-1)
        with open('./export/' + filename + '_dataout.txt', 'w') as f:
            for item in weight_txt:
                f.write("%s\n" % item)  
#revised by chengquan

def hook_normin(module, input, result, filename):
    # 获取中间层的权重和激活值
    if module.mode == "int":
        if not torch.is_tensor(module.scalein):
            weight = module.scalein
            weight_txt = weight.reshape(-1)
            #activation = output.data.cpu().numpy()
            # 将权重和激活值保存到二进制文件中
            with open('./export/' + filename + '_scalein.bin', 'wb') as f:
                f.write(weight.tobytes())

            with open('./export/' + filename + '_scalein.txt', 'w') as f:
                for item in weight_txt:
                    f.write("%s\n" % item)

        if not torch.is_tensor(module.scaleout):
            weight = module.scaleout
            weight_txt = weight.reshape(-1)
            #activation = output.data.cpu().numpy()
            # 将权重和激活值保存到二进制文件中
            with open('./export/' + filename + '_scaleout.bin', 'wb') as f:
                f.write(weight.tobytes())

            with open('./export/' + filename + '_scaleout.txt', 'w') as f:
                for item in weight_txt:
                    f.write("%s\n" % item)

        if not torch.is_tensor(module.weightdata):
            weight = module.weightdata
            weight_txt = weight.reshape(-1)
            with open('./export/' + filename + '_weight.txt', 'w') as f:
                for item in weight_txt:
                    f.write("%s\n" % item)

        if not torch.is_tensor(module.biasdata):
            weight = module.biasdata
            weight_txt = weight.reshape(-1)
            with open('./export/' + filename + '_bias.txt', 'w') as f:
                for item in weight_txt:
                    f.write("%s\n" % item)  





class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 quant=False,
                 calibrate=False,
                 cfg=None,
                 blknum=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = QLinear(dim,
                           dim * 3,
                           bias=qkv_bias,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_W,
                           calibration_mode=cfg.CALIBRATION_MODE_W,
                           observer_str=cfg.OBSERVER_W,
                           quantizer_str=cfg.QUANTIZER_W)
        
        hook_attn_qkv = functools.partial(hook_linear, filename="attn_qkv_"+str(blknum))
        self.qkv.register_forward_hook(hook_attn_qkv) #添加hook函数     

        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        
        hook_attn_qact1 = functools.partial(hook_actin, filename="attn_qact1_"+str(blknum))
        self.qact1.register_forward_hook(hook_attn_qact1) #添加hook函数          

        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        
        hook_attn_qact2 = functools.partial(hook_actin, filename="attn_qact2_"+str(blknum))
        self.qact2.register_forward_hook(hook_attn_qact2) #添加hook函数             

        self.proj = QLinear(dim,
                            dim,
                            quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W)
        
        hook_attn_proj = functools.partial(hook_linear, filename="attn_proj_"+str(blknum))
        self.proj.register_forward_hook(hook_attn_proj) #添加hook函数          

        self.qact3 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        
        hook_attn_qact3 = functools.partial(hook_actin, filename="attn_qact3_"+str(blknum))
        self.qact3.register_forward_hook(hook_attn_qact3) #添加hook函数    


        self.qact_attn1 = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)
        
        hook_qact_attn1 = functools.partial(hook_actin, filename="attn_qact_attn1_"+str(blknum))
        self.qact_attn1.register_forward_hook(hook_qact_attn1) #添加hook函数         

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.log_int_softmax = QIntSoftmax(
            log_i_softmax=cfg.INT_SOFTMAX,
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_S,
            calibration_mode=cfg.CALIBRATION_MODE_S,
            observer_str=cfg.OBSERVER_S,
            quantizer_str=cfg.QUANTIZER_S)
        

    def forward(self, x):
        B, N, C = x.shape
        x = self.qkv(x)
        x = self.qact1(x)
        qkv = x.reshape(B, N, 3, self.num_heads,
                        C // self.num_heads).permute(2, 0, 3, 1, 4)  # (BN33)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.qact_attn1(attn)
        attn = self.log_int_softmax(attn, self.qact_attn1.quantizer.scale)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.qact2(x)
        x = self.proj(x)
        x = self.qact3(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 quant=False,
                 calibrate=False,
                 cfg=None,
                 blknum=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #revised by chengquan
        hook_block_norm1 = functools.partial(hook_normin, filename="block_norm1_b"+str(blknum))
        self.norm1.register_forward_hook(hook_block_norm1) #添加hook函数
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        #revised by chengquan
        hook_block_qact1 = functools.partial(hook_actin, filename="block_qact1_b"+str(blknum))
        self.qact1.register_forward_hook(hook_block_qact1) #添加hook函数

        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              cfg=cfg,
                              blknum=blknum)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                          observer_str=cfg.OBSERVER_A_LN,
                          quantizer_str=cfg.QUANTIZER_A_LN)
        #revised by chengquan
        hook_block_qact2 = functools.partial(hook_actin, filename="block_qact2_b"+str(blknum))
        self.qact2.register_forward_hook(hook_block_qact2) #添加hook函数        


        self.norm2 = norm_layer(dim)
        #revised by chengquan
        hook_block_norm2 = functools.partial(hook_normin, filename="block_norm2_b"+str(blknum))
        self.norm2.register_forward_hook(hook_block_norm2) #添加hook函数
        self.qact3 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        
        #revised by chengquan
        hook_block_qact3 = functools.partial(hook_actin, filename="block_qact3_b"+str(blknum))
        self.qact3.register_forward_hook(hook_block_qact3) #添加hook函数        

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop,
                       quant=quant,
                       calibrate=calibrate,
                       cfg=cfg,
                       blknum=blknum)
        
        self.qact4 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                          observer_str=cfg.OBSERVER_A_LN,
                          quantizer_str=cfg.QUANTIZER_A_LN)
        
        #revised by chengquan
        hook_block_qact4 = functools.partial(hook_actin, filename="block_qact4_b"+str(blknum))
        self.qact4.register_forward_hook(hook_block_qact4) #添加hook函数     

    def forward(self, x, last_quantizer=None):
        x = self.qact2(x + self.drop_path(
            self.attn(
                self.qact1(self.norm1(x, last_quantizer,
                                      self.qact1.quantizer)))))
        x = self.qact4(x + self.drop_path(
            self.mlp(
                self.qact3(
                    self.norm2(x, self.qact2.quantizer,
                               self.qact3.quantizer)))))
        return x






class VisionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 hybrid_backbone=None,
                 norm_layer=None,
                 quant=False,
                 calibrate=False,
                 input_quant=False,
                 cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.cfg = cfg
        self.input_quant = input_quant
        if input_quant:
            self.qact_input = QAct(quant=quant,
                                   calibrate=calibrate,
                                   bit_type=cfg.BIT_TYPE_A,
                                   calibration_mode=cfg.CALIBRATION_MODE_A,
                                   observer_str=cfg.OBSERVER_A,
                                   quantizer_str=cfg.QUANTIZER_A)
            #revised by chengquan
            hook_qact_input = functools.partial(hook_actin, filename="qact_input")
            self.qact_input.register_forward_hook(hook_qact_input) #添加hook函数

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(img_size=img_size,
                                          patch_size=patch_size,
                                          in_chans=in_chans,
                                          embed_dim=embed_dim,
                                          quant=quant,
                                          calibrate=calibrate,
                                          cfg=cfg)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.qact_embed = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)
        #revised by chengquan
        hook_qact_embed = functools.partial(hook_actin, filename="qact_embed")
        self.qact_embed.register_forward_hook(hook_qact_embed) #添加hook函数       

        self.qact_pos = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)
        
        #revised by chengquan
        hook_qact_pos = functools.partial(hook_actin, filename="qact_pos")
        self.qact_pos.register_forward_hook(hook_qact_pos) #添加hook函数           

        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                          observer_str=cfg.OBSERVER_A_LN,
                          quantizer_str=cfg.QUANTIZER_A_LN)
        #revised by chengquan
        hook_qact1 = functools.partial(hook_actin, filename="qact1")
        self.qact1.register_forward_hook(hook_qact1) #添加hook函数    

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  quant=quant,
                  calibrate=calibrate,
                  cfg=cfg,
                  blknum=i) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        hook_fn_norm2 = functools.partial(hook_normin, filename="fnorm")
        self.norm.register_forward_hook(hook_fn_norm2) #添加hook函数

        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        #revised by chengquan
        hook_qact2 = functools.partial(hook_actin, filename="qact2")
        self.qact2.register_forward_hook(hook_qact2) #添加hook函数   

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict([
                    ('fc', nn.Linear(embed_dim, representation_size)),
                    ('act', nn.Tanh()),
                ]))
        else:
            self.pre_logits = nn.Identity()

        #hook_special = functools.partial(hook_sp, filename="cls_token")
        #self.pre_logits.register_forward_hook(hook_special) #添加hook函数    

        # Classifier head
        self.head = (QLinear(self.num_features,
                             num_classes,
                             quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_W,
                             calibration_mode=cfg.CALIBRATION_MODE_W,
                             observer_str=cfg.OBSERVER_W,
                             quantizer_str=cfg.QUANTIZER_W)
                     if num_classes > 0 else nn.Identity())
        
        #revised by chengquan
        hook_head_linear = functools.partial(hook_linear, filename="head")
        self.head.register_forward_hook(hook_head_linear) #添加hook函数        

        self.act_out = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)
        #revised by chengquan
        hook_act_out = functools.partial(hook_actin, filename="act_out")
        self.act_out.register_forward_hook(hook_act_out) #添加hook函数

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = (nn.Linear(self.embed_dim, num_classes)
                     if num_classes > 0 else nn.Identity())

    def model_quant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = True
            if self.cfg.INT_NORM:
                if type(m) in [QIntLayerNorm]:
                    m.mode = 'int'

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = False

    def forward_features(self, x):
        B = x.shape[0]

        if self.input_quant:
            x = self.qact_input(x)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        
        clk_tokenss = cls_tokens[0].cpu().numpy()
        clk_tokenss = clk_tokenss.reshape(-1)
        with open('./export/cls_token.txt', 'w') as f:
            for item in clk_tokenss:
                f.write("%s\n" % item)   

        x = torch.cat((cls_tokens, x), dim=1)
        x = self.qact_embed(x)
        x = x + self.qact_pos(self.pos_embed)

        pos_embedd = self.pos_embed.cpu().numpy()
        pos_embedd = pos_embedd.reshape(-1)       
        with open('./export/pos_embed.txt', 'w') as f:
            for item in pos_embedd:
                f.write("%s\n" % item)   

        x = self.qact1(x)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            last_quantizer = self.qact1.quantizer if i == 0 else self.blocks[
                i - 1].qact4.quantizer
            x = blk(x, last_quantizer)

        x = self.norm(x, self.blocks[-1].qact4.quantizer,
                      self.qact2.quantizer)[:, 0]
        x = self.qact2(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = self.act_out(x)
        return x


def deit_tiny_patch16_224(pretrained=False,
                          quant=False,
                          calibrate=False,
                          cfg=None,
                          **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(QIntLayerNorm, eps=1e-6),
        quant=quant,
        calibrate=calibrate,
        input_quant=True,
        cfg=cfg,
        **kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=
            'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
            map_location='cpu',
            check_hash=True,
        )
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


def deit_small_patch16_224(pretrained=False,
                           quant=False,
                           calibrate=False,
                           cfg=None,
                           **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=384,
                              depth=12,
                              num_heads=6,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=True,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=
            'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
            map_location='cpu',
            check_hash=True)
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


def deit_base_patch16_224(pretrained=False,
                          quant=False,
                          calibrate=False,
                          cfg=None,
                          **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=True,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=
            'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
            map_location='cpu',
            check_hash=True)
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


def vit_base_patch16_224(pretrained=False,
                         quant=False,
                         calibrate=False,
                         cfg=None,
                         **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=False,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        url = 'https://storage.googleapis.com/vit_models/augreg/' + \
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'

        load_weights_from_npz(model, url, check_hash=True)
    return model


def vit_large_patch16_224(pretrained=False,
                          quant=False,
                          calibrate=False,
                          cfg=None,
                          **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=False,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        url = 'https://storage.googleapis.com/vit_models/augreg/' + \
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'

        load_weights_from_npz(model, url, check_hash=True)
    return model

import sys
import torch
from PIL import Image
import numpy as np


def load_act(out_file):
    outdata = []
    with open(out_file, 'r') as file:
        for line in file:
            line = line.strip()
            outdata.append(float(line))
    outdata = np.array(outdata)
    return outdata

def load_weight_and_bias(weight_file, bias_file=None):
    bias = []
    wt = []
    if bias_file != None:
        with open(bias_file, 'r') as file:
            for line in file:
                line = line.strip()
                bias.append(float(line))
        bias = np.array(bias)

    with open(weight_file, 'r') as file:
        for line in file:
            line = line.strip()
            wt.append(float(line))
    wt = np.array(wt)
    return wt, bias

def load_scale_and_zp(scale_file, zp_file):
    scale = []
    with open(scale_file, 'r') as file:
        for line in file:
            line = line.strip()
            scale.append(float(line))
    scale = np.array(scale)
    zp = []
    with open(zp_file, 'r') as file:
        for line in file:
            line = line.strip()
            zp.append(float(line))
    zp = np.array(zp)
    return scale, zp

def compare_arrays(A, B, min_d):
    count = np.sum(np.abs(A - B) >= min_d)
    return count


def linear(x, weight, bias=None):
    if bias is None:
        return np.dot(x, weight.T)
    else:
        return np.dot(x, weight.T) + bias

def get_MN(x):
	bit = 8
	N = np.clip(bit - 1 - np.floor(np.log2(x)), 0, 31).astype(int)
	M = np.clip(np.floor(x * 2**N), 0, 2**bit - 1).astype(int)
	return M, N

def convolve(image, weights, bias, stride=1, padding=0):
    # 获取输入数据的形状和卷积核的形状
    (image_channels, image_height, image_width) = image.shape
    (output_channels, weight_channels, weight_height, weight_width) = weights.shape
    # 计算输出特征图的形状
    output_height = int((image_height + 2 * padding - weight_height) / stride) + 1
    output_width = int((image_width + 2 * padding - weight_width) / stride) + 1
    # 对输入数据进行填充
    image_pad = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), 'constant')
    # 初始化输出特征图
    output = np.zeros((output_channels, output_height, output_width))
    # 进行卷积运算
    for oc in range(output_channels):
        for ic in range(image_channels):
            for oh in range(output_height):
                for ow in range(output_width):
                    for kh in range(weight_height):
                        for kw in range(weight_width):
                            ih = oh * stride + kh
                            iw = ow * stride + kw
                            output[oc, oh, ow] += image_pad[ic, ih, iw] * weights[oc, ic, kh, kw]
        output[oc] += bias[oc]
    return output


def log_round(x):
    x_log_floor = np.floor(np.log2(x))
    big = x_log_floor
    extra_mask = (x - 2**big) >= 2**(big - 1)
    big[extra_mask] = big[extra_mask] + 1
    return big

def int_softmax(x, scaling_factor):
    def int_polynomial(x_int, scaling_factor):
        coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
        coef[1] /= coef[0]
        coef[2] /= coef[0]
        b_int = np.floor(coef[1] / scaling_factor)
        c_int = np.floor(coef[2] / scaling_factor**2)
        z = x_int + b_int
        z = x_int * z
        z = z + c_int
        scaling_factor = coef[0] * scaling_factor**2
        return z, scaling_factor

    def int_exp_numpy(x_int, scaling_factor):
        x0 = -0.6931  # -ln2
        n = 30  # sufficiently large integer
        x0_int = np.floor(x0 / scaling_factor)
        x_int = np.maximum(x_int, n * x0_int)
        q = np.floor(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
        exp_int = np.clip(np.floor(exp_int * 2**(n - q)), a_min=0, a_max=None)
        scaling_factor = exp_scaling_factor / 2**n
        return exp_int, scaling_factor

    x_int = x / scaling_factor
    # 对整数部分 x_int 按照最后一个维度求最大值，并将其与 x_int 作差得到整数部分的范围在 [0, scaling_factor) 之内的数
    x_int_max = np.max(x_int, axis=-1, keepdims=True)
    x_int = x_int - x_int_max
    # 使用一个名为 int_exp 的函数对整数部分 x_int 进行指数化处理，得到指数部分 exp_int 和缩放因子 exp_scaling_factor
    exp_int, exp_scaling_factor = int_exp_numpy(x_int, scaling_factor)
    # 对指数部分 exp_int 按照最后一个维度求和，得到一个指数和 exp_int_sum
    exp_int_sum = np.sum(exp_int, axis=-1, keepdims=True)
    return exp_int, exp_int_sum

def log_softmax(x, scale):
    bits = 4
    exp_int, exp_int_sum = int_softmax(x, scale)
    softmax_out = np.round(exp_int_sum / exp_int)
    rounds = log_round(softmax_out)
    mask = rounds >= 2**bits
    qlog = np.clip(rounds, 0, 2**bits - 1)
    deq_softmax = 2**(-qlog)
    deq_softmax[mask] = 0
    return deq_softmax

def verify_linear(layer, d_input, wt, bias, scale, zp, pscale, pzp, outd, expand  ):
	#[197,384]
	actlist = d_input
	d_pscale = []
	d_pzp    = []
	d_scale = []
	d_zp    = []
	d_wt    = []
	d_bias  = []
	outdata = []
	d_scale, d_zp = load_scale_and_zp(scale, zp)
	d_wt, d_bias = load_weight_and_bias(wt, bias)
	d_pscale, d_pzp = load_scale_and_zp(pscale, pzp)
	d_wt = d_wt.reshape(int(actlist.shape[1] * expand), actlist.shape[1])
	for i in range(int(actlist.shape[1] * expand)):
	    d_wt[i] = (d_wt[i] - d_zp[i]) * d_scale[i]
	actlist = linear(actlist, d_wt, d_bias)
	actlist = actlist / d_pscale + d_pzp

	outdata = load_act(outd)
	outdata = outdata.reshape(actlist.shape)
	print(layer, end='')
	print(" shape:", outdata.shape)
	min_data = 1.0
	error_count = compare_arrays(actlist, outdata, min_data)
	print(layer, end='')
	print(" total samples:%d, total error:%d" %(actlist.size, error_count))
	return outdata, d_pscale, d_pzp

def gelu(x):
    """GELU activation function."""
    cdf = 0.5 * (1.0 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))))
    return x * cdf

def verify_linear2(layer, d_input, wt, bias, scale, zp, pscale, pzp, outd, expand  ):
	#[197,384]
	actlist = d_input
	d_pscale = []
	d_pzp    = []
	d_scale = []
	d_zp    = []
	d_wt    = []
	d_bias  = []
	outdata = []
	d_scale, d_zp = load_scale_and_zp(scale, zp)
	d_wt, d_bias = load_weight_and_bias(wt, bias)
	d_pscale, d_pzp = load_scale_and_zp(pscale, pzp)
	d_wt = d_wt.reshape(actlist.shape[1] * expand, actlist.shape[1])
	for i in range(actlist.shape[1] * expand):
	    d_wt[i] = (d_wt[i] - d_zp[i]) * d_scale[i]
	actlist = linear(actlist, d_wt, d_bias)
	actlist = gelu(actlist)
	actlist = actlist / d_pscale + d_pzp

	outdata = load_act(outd)
	outdata = outdata.reshape(actlist.shape)
	print(layer, end='')
	print(" shape:", outdata.shape)
	min_data = 1.0
	error_count = compare_arrays(actlist, outdata, min_data)
	print(layer, end='')
	print(" total samples:%d, total error:%d" %(actlist.size, error_count))
	return outdata, d_pscale, d_pzp


	
def verify_layernorm(layer, d_input, weight, bias, scalein, scaleout, d_scale, d_zp, out_d):
    d_wt = []
    d_bias   = []
    in_scale = []
    out_scale = []
    outdata = []
    x = d_input
    d_wt, d_bias = load_weight_and_bias(weight, bias)
    in_scale, out_scale =  load_scale_and_zp(scalein, scaleout)

    channel_nums = x.shape[-1]
    in_scale = in_scale.reshape(1, 1, -1)
    out_scale = out_scale.reshape(1, 1, -1)
    x_q = (x / in_scale).round()
    in_scale1 = in_scale.min()
    in_scale_mask = (in_scale / in_scale1).round()
    x_q = x_q * in_scale_mask
    # 计算均值和标准差
    mean_x_q = np.mean(x_q, axis=-1) * in_scale1
    std_x_q = (in_scale1 / channel_nums) * np.sqrt(channel_nums * np.sum(x_q**2, axis=-1) - np.sum(x_q, axis=-1)**2)
    # 计算矩阵乘积
    A = (in_scale1 / std_x_q).reshape(-1, 1) * d_wt.reshape(1, -1) / out_scale
    #print(A.shape)
    A_sign = np.sign(A)
    M, N = get_MN(np.abs(A))
    B = ((d_bias.reshape(1, -1) - (mean_x_q / std_x_q).reshape(-1, 1) * d_wt.reshape(1, -1)) / out_scale * 2**N).round()
    x_q = ((A_sign * M * x_q + B) / 2**N).round()
    x = x_q * out_scale
    x2 = x.reshape(d_input.shape)
    x2 = x2 / d_scale + d_zp
    print(layer, end='')
    print(" shape:", x2.shape)
    #output
    with open(out_d, 'r') as file:
        for line in file:
            line = line.strip()
            outdata.append(float(line))
    outdata = np.array(outdata)
    outdata = outdata.reshape(x2.shape)
    min_data = 1.0
    error_count = compare_arrays(x2, outdata, min_data)
    print(layer, end='')
    print(" total samples:%d, total error:%d" %(x2.size, error_count))
    return outdata	
	

act_list      = []
act_scale     = []
act_zp        = []
wt_list       = []
wt_scale      = []
wt_zp         = []
wt_bias       = []
outdata       = []
#layer #1

#activation #1    1.input-channel 2.size_y 3.size_x
qact_input_size = [3,224,224]
qact_input       = "../export/qact_input_int8.txt"
qact_input_scale = "../export/qact_input_scale_cfg.txt"
qact_input_zp    = "../export/qact_input_zeropoint_cfg.txt"


#weight #1      1.output-channel 2.input-channel 3.size_y 4.size_x  
conv_in_size  = [384,3,16,16]
conv_in       = "../export/conv_in_int8.txt"
conv_in_scale = "../export/conv_in_scale_cfg.txt"
conv_in_zp    = "../export/conv_in_zeropoint_cfg.txt"
conv_in_bias  = "../export/conv_in_bias.txt"

#output #1
output_size = [384,14,14]
output_file = "../export/conv_in_dataout.txt"


#act
act_scale, act_zp = load_scale_and_zp(qact_input_scale, qact_input_zp)
act_list = load_act(qact_input)
act_list = act_list.reshape(qact_input_size[0],qact_input_size[1],qact_input_size[2])
act_list = act_list - act_zp
#print(act_list[0][0][0])
#print(act_list[0][0][1])

#wt
wt_scale, wt_zp = load_scale_and_zp(conv_in_scale, conv_in_zp)
wt_list, wt_bias = load_weight_and_bias(conv_in, conv_in_bias)
wt_list = wt_list.reshape(conv_in_size[0],conv_in_size[1],conv_in_size[2],conv_in_size[3])

for i in range(conv_in_size[0]):
    wt_list[i] = wt_list[i] - wt_zp[i]
#print(wt_list[0][0][0][0])
#print(wt_list[0][0][0][1])

#op 14x14x384
wt_bias = wt_bias / act_scale
for i in range(conv_in_size[0]):
    wt_bias[i] = wt_bias[i] / wt_scale[i]
wt_bias = np.round(wt_bias)
output1 = convolve(act_list, wt_list, wt_bias, stride=16, padding=0)

output1 = output1 * act_scale
for i in range(conv_in_size[0]):
    output1[i] = output1[i] * wt_scale[i]
print("#########Conv2D")
print("Image shape:", act_list.shape)
print("Weight#1 shape:", wt_list.shape)
print("Out#1 shape:", output1.shape)
################################################## 
act_list      = []
act_scale     = []
act_zp        = []
outdata       = []

#activation #1    1.input-channel 2.size_y 3.size_x
qact_input_size = [384,196]
qact_input_scale = "../export/layer_out0_scale_cfg.txt"
qact_input_zp    = "../export/layer_out0_zeropoint_cfg.txt"

output_size = [196,384]
output_file = "../export/layer_out0_int8.txt"
###################PatchEmbed####################
act_list = output1.reshape(384,-1)
act_list = act_list.T  
print("Act#1 shape:", act_list.shape)

act_scale, act_zp = load_scale_and_zp(qact_input_scale, qact_input_zp)

#output
outdata = load_act(output_file)
outdata = outdata.reshape(act_list.shape)

act_list = act_list/act_scale + act_zp
#act_list = np.round(act_list/act_scale + act_zp)

min_data = 1.0
error_count = compare_arrays(act_list, outdata, min_data)
print("Layer#1 total samples:%d, total error:%d" %(act_list.size, error_count))
# 在此处处理浮点数数组
# print(min_data)
# print(act_list[0][0])
# print(act_list[0][1])

##############cls token######################
print("#########cls_token")
act_list = outdata
cls_size = [1,384]
cls_file = "../export/cls_token.txt"
cls_data = []
cls_concat = []
#output
with open(cls_file, 'r') as file:
    for line in file:
        line = line.strip()
        cls_data.append(float(line))
cls_data = np.array(cls_data)
cls_data = cls_data/act_scale + act_zp
cls_data = np.round(cls_data)

cls_concat = np.concatenate((cls_data.reshape(1, -1), act_list), axis=0)
print("cls concat shape:", cls_concat.shape)


qact_embed_scale = "../export/qact_embed_scale_cfg.txt"
qact_embed_zp    = "../export/qact_embed_zeropoint_cfg.txt"
cls_scale      = []
cls_zp         = []

cls_scale, cls_zp = load_scale_and_zp(qact_embed_scale, qact_embed_zp)

cls_concat = ((cls_concat - act_zp) * act_scale) / cls_scale + cls_zp

output_size = [197,384]
output_file = "../export/qact_embed_int8.txt"
#output
outdata = []
outdata = load_act(output_file)
outdata = outdata.reshape(cls_concat.shape)

min_data = 1.0
error_count = compare_arrays(cls_concat, outdata, min_data)
print("cls_layer total samples:%d, total error:%d" %(cls_concat.size, error_count))

###############pos_embed######################
print("#########pos_embed")
pos_file             = "../export/pos_embed.txt"
qact_pos_scale_file  = "../export/qact_pos_scale_cfg.txt"
pos_scale_file = "../export/qact1_scale_cfg.txt"
pos_zp_file    = "../export/qact1_zeropoint_cfg.txt"

actlist = outdata
pos_scale  = []
pos_zp     = []
pos_data   = []
qact_pos_scale = []
with open(qact_pos_scale_file, 'r') as file:
    for line in file:
        line = line.strip()
        qact_pos_scale.append(float(line))
qact_pos_scale = np.array(qact_pos_scale)

pos_scale, pos_zp = load_scale_and_zp(pos_scale_file, pos_zp_file)

#
pos_data = load_act(pos_file)
pos_data = pos_data.reshape(cls_concat.shape)

#pos_data / pos_scale / cls_scale * pos_scale

pos_data = np.round(pos_data / qact_pos_scale) / cls_scale * qact_pos_scale

actlist = (pos_data + actlist - cls_zp) * cls_scale 

#actlist = (actlist - cls_zp) * cls_scale + pos_data
#使用broadcasting将1x384的数组复制为197x384的数组
pos_scale_broadcasted = np.tile(pos_scale, (pos_data.shape[0], 1))
actlist = actlist / pos_scale_broadcasted + pos_zp

output_size = [197,384]
output_file = "../export/qact1_int8.txt"
outdata = []
with open(output_file, 'r') as file:
    for line in file:
        line = line.strip()
        outdata.append(float(line))
outdata = np.array(outdata)
outdata = outdata.reshape(actlist.shape)

print("pos_embed shape:", actlist.shape)
min_data = 1.0
error_count = compare_arrays(actlist, outdata, min_data)
print("pos_embed total samples:%d, total error:%d" %(actlist.size, error_count))

###############attention block######################
#norm -> QAct1 -> Attention -> QAct2 -> norm -> QAct3 -> MLP -> QAct4 
block_file_head  = "0"

print("#########block:",block_file_head)
qact1_file       = "../export/block_qact1_b"+block_file_head+"_int8.txt"
qact1_scale_file = "../export/block_qact1_b"+block_file_head+"_scale_cfg.txt"
qact1_zp_file    = "../export/block_qact1_b"+block_file_head+"_zeropoint_cfg.txt"

actlist = outdata
outdata = []
qact1_scale = []
qact1_zp = []

qact1_scale, qact1_zp = load_scale_and_zp(qact1_scale_file, qact1_zp_file)


actlist = (actlist - pos_zp) * pos_scale_broadcasted

#norm1_file_file     = "../export/block_norm1_b"+block_file_head+"_weight.txt"
norm1_weight_file   = "../export/block_norm1_b"+block_file_head+"_weight.txt"
norm1_bias_file     = "../export/block_norm1_b"+block_file_head+"_bias.txt"
norm1_scalein_file  = "../export/block_norm1_b"+block_file_head+"_scalein.txt"
norm1_scaleout_file = "../export/block_norm1_b"+block_file_head+"_scaleout.txt"

resAdd1 = actlist

outdata = verify_layernorm("norm1", actlist, norm1_weight_file, norm1_bias_file, norm1_scalein_file, norm1_scaleout_file, qact1_scale, qact1_zp, qact1_file)

#######attn_qkv_########################
#qkv: input 197x384 output 197x1152    weight: 384x1152  bias:1152  
#dim: 384
actlist = outdata
qkv_weight_file = "../export/attn_qkv_"+block_file_head+"_wint.txt"
qkv_bias_file   = "../export/attn_qkv_"+block_file_head+"_bias.txt"
qkv_scale_file = "../export/attn_qkv_"+block_file_head+"_scale_cfg.txt"
qkv_zp_file   = "../export/attn_qkv_"+block_file_head+"_zeropoint_cfg.txt"

qkv_pscale_file  = "../export/attn_qact1_"+block_file_head+"_scale_cfg.txt"
qkv_pzp_file     = "../export/attn_qact1_"+block_file_head+"_zeropoint_cfg.txt"


qkv_pscale = []
qkv_pzp   = []
output_file = "../export/attn_qact1_"+block_file_head+"_int8.txt"

actlist = (actlist - qact1_zp) * qact1_scale
outdata, qkv_pscale, qkv_pzp= verify_linear("qkv", actlist, qkv_weight_file, qkv_bias_file, qkv_scale_file, qkv_zp_file, qkv_pscale_file, qkv_pzp_file, output_file, 3)
#######transpose 
#197 6 3 64
num_heads = 6
outdata = outdata.reshape(197, 3, int(num_heads), int(384/num_heads)).transpose(1,2,0,3)
print("transpose size is", outdata.shape) 
scale = (384 // num_heads) ** (-0.5)

outdata = (outdata - qkv_pzp) * qkv_pscale
q = outdata[0]
k = outdata[1].transpose(0,2,1)
v = outdata[2]
#print(q.shape)
#print(k.shape)
attn = q @ k * scale

qact_attn1_file       = "../export/attn_qact_attn1_"+block_file_head+"_int8.txt" 
qact_attn1_scale_file = "../export/attn_qact_attn1_"+block_file_head+"_scale_cfg.txt"
qact_attn1_zp_file    = "../export/attn_qact_attn1_"+block_file_head+"_zeropoint_cfg.txt"

qact_attn1_scale = []
qact_attn1_zp    = []
qact_attn1       = []

qact_attn1_scale, qact_attn1_zp = load_scale_and_zp(qact_attn1_scale_file, qact_attn1_zp_file)
qact_attn1 = load_act(qact_attn1_file)
qact_attn1 = qact_attn1.reshape(attn.shape)

attn_cmp = attn / qact_attn1_scale + qact_attn1_zp

print("attn shape:", attn.shape)
min_data = 1.0
error_count = compare_arrays(attn_cmp, qact_attn1, min_data)
print("attn total samples:%d, total error:%d" %(attn_cmp.size, error_count))

qact_attn1 = (qact_attn1 - qact_attn1_zp) * qact_attn1_scale
attn = log_softmax(qact_attn1, qact_attn1_scale)

attn = (attn @ v).transpose(1,0,2)
print("q*k_T*v shape:", attn.shape)
attn = attn.reshape(197,384)

attn_qact2_file       = "../export/attn_qact2_"+block_file_head+"_int8.txt" 
attn_qact2_scale_file = "../export/attn_qact2_"+block_file_head+"_scale_cfg.txt"
attn_qact2_zp_file    = "../export/attn_qact2_"+block_file_head+"_zeropoint_cfg.txt"

attn_qact2_scale = []
attn_qact2_zp    = []
attn_qact2       = []

attn_qact2_scale, attn_qact2_zp = load_scale_and_zp(attn_qact2_scale_file, attn_qact2_zp_file)
attn_qact2 = load_act(attn_qact2_file)
attn_qact2 = attn_qact2.reshape(attn.shape)

attn = attn / attn_qact2_scale + attn_qact2_zp
min_data = 1.0
error_count = compare_arrays(attn, attn_qact2, min_data)
print("q*k_T*v total samples:%d, total error:%d" %(attn.size, error_count))

##############################
attn_proj_scale_file = "../export/attn_proj_"+block_file_head+"_scale_cfg.txt"
attn_proj_zp_file    = "../export/attn_proj_"+block_file_head+"_zeropoint_cfg.txt"
attn_proj_wt_file    = "../export/attn_proj_"+block_file_head+"_wint.txt"
attn_proj_bias_file  = "../export/attn_proj_"+block_file_head+"_bias.txt"
attn_proj_pscale_file = "../export/attn_qact3_"+block_file_head+"_scale_cfg.txt"
attn_proj_pzp_file    = "../export/attn_qact3_"+block_file_head+"_zeropoint_cfg.txt"
output_file           = "../export/attn_qact3_"+block_file_head+"_int8.txt"

#[197,384]
actlist = (attn_qact2 - attn_qact2_zp) * attn_qact2_scale
outdata = []
proj_pscale = []
proj_pzp = []
outdata, proj_pscale, proj_pzp = verify_linear("proj", actlist, attn_proj_wt_file, attn_proj_bias_file, attn_proj_scale_file, attn_proj_zp_file, attn_proj_pscale_file, attn_proj_pzp_file, output_file, 1)
##############################

qact2_scale_file  = "../export/block_qact2_b"+block_file_head+"_scale_cfg.txt"
qact2_zp_file     = "../export/block_qact2_b"+block_file_head+"_zeropoint_cfg.txt"
qact2_file        = "../export/block_qact2_b"+block_file_head+"_int8.txt"
qact2_scale = []
qact2_zp    = []
qact2_scale, qact2_zp = load_scale_and_zp(qact2_scale_file, qact2_zp_file)

outdata = (outdata - proj_pzp) * proj_pscale
#for i in range(384):
#	print(resAdd1[0][i])
outdata = resAdd1 + outdata

#使用broadcasting将1x384的数组复制为197x384的数组
qact2_scale_broadcasted = np.tile(qact2_scale, (outdata.shape[0], 1))
outdata = np.round(outdata / qact2_scale_broadcasted) + qact2_zp
#outdata = (outdata - qact2_zp) * qact2_scale_broadcasted

#veri = load_act(qact2_file)
#veri = veri.reshape(outdata.shape)
#min_data = 1.0
#error_count = compare_arrays(veri, outdata, min_data)
#print(" total samples:%d, total error:%d" %(veri.size, error_count))
outdata = (outdata - qact2_zp) * qact2_scale_broadcasted

resAdd2 = outdata
###########################norm 2    
norm2_weight_file   = "../export/block_norm2_b"+block_file_head+"_weight.txt"
norm2_bias_file     = "../export/block_norm2_b"+block_file_head+"_bias.txt"
norm2_scalein_file  = "../export/block_norm2_b"+block_file_head+"_scalein.txt"
norm2_scaleout_file = "../export/block_norm2_b"+block_file_head+"_scaleout.txt"
qact3_file       = "../export/block_qact3_b"+block_file_head+"_int8.txt"
qact3_scale_file = "../export/block_qact3_b"+block_file_head+"_scale_cfg.txt"
qact3_zp_file    = "../export/block_qact3_b"+block_file_head+"_zeropoint_cfg.txt"

actlist = outdata
qact3_scale = []
qact3_zp = []
qact3_scale, qact3_zp = load_scale_and_zp(qact3_scale_file, qact3_zp_file)
actlist = outdata
outdata = verify_layernorm("norm2", actlist, norm2_weight_file, norm2_bias_file, norm2_scalein_file, norm2_scaleout_file, qact3_scale, qact3_zp, qact3_file)
####################################
#mlp_fc1
actlist = (outdata - qact3_zp) * qact3_scale

mlpfc1_weight_file = "../export/mlp_fc1"+block_file_head+"_wint.txt"
mlpfc1_bias_file   = "../export/mlp_fc1"+block_file_head+"_bias.txt"
mlpfc1_scale_file  = "../export/mlp_fc1"+block_file_head+"_scale_cfg.txt"
mlpfc1_zp_file     = "../export/mlp_fc1"+block_file_head+"_zeropoint_cfg.txt"

mlp_qact1_file       = "../export/mlp_qact1"+block_file_head+"_int8.txt"
mlp_qact1_scale_file = "../export/mlp_qact1"+block_file_head+"_scale_cfg.txt"
mlp_qact1_zp_file    = "../export/mlp_qact1"+block_file_head+"_zeropoint_cfg.txt"


outdata, mlpfc1_scale, mlpfc2_zp= verify_linear2("mlp_fc1", actlist, mlpfc1_weight_file, mlpfc1_bias_file, mlpfc1_scale_file, mlpfc1_zp_file, mlp_qact1_scale_file, mlp_qact1_zp_file, mlp_qact1_file, 4)


#mlp_fc2
mlpfc2_weight_file = "../export/mlp_fc2"+block_file_head+"_wint.txt"
mlpfc2_bias_file   = "../export/mlp_fc2"+block_file_head+"_bias.txt"
mlpfc2_scale_file  = "../export/mlp_fc2"+block_file_head+"_scale_cfg.txt"
mlpfc2_zp_file     = "../export/mlp_fc2"+block_file_head+"_zeropoint_cfg.txt"

mlp_qact2_file       = "../export/mlp_qact2"+block_file_head+"_int8.txt"
mlp_qact2_scale_file = "../export/mlp_qact2"+block_file_head+"_scale_cfg.txt"
mlp_qact2_zp_file    = "../export/mlp_qact2"+block_file_head+"_zeropoint_cfg.txt"


outdata = (outdata - mlpfc2_zp) * mlpfc1_scale
outdata, mlpfc2_scale, mlpfc2_zp= verify_linear("mlp_fc2", outdata, mlpfc2_weight_file, mlpfc2_bias_file, mlpfc2_scale_file, mlpfc2_zp_file, mlp_qact2_scale_file, mlp_qact2_zp_file, mlp_qact2_file, 0.25)
#for i in range(384):
#	print(outdata[0][i])
outdata = (outdata - mlpfc2_zp) * mlpfc2_scale

qact4_scale_file  = "../export/block_qact4_b"+block_file_head+"_scale_cfg.txt"
qact4_zp_file     = "../export/block_qact4_b"+block_file_head+"_zeropoint_cfg.txt"
qact4_file        = "../export/block_qact4_b"+block_file_head+"_int8.txt"
qact4_scale = []
qact4_zp    = []
qact4_scale, qact4_zp = load_scale_and_zp(qact4_scale_file, qact4_zp_file)


outdata = resAdd2 + outdata

#使用broadcasting将1x384的数组复制为197x384的数组
qact4_scale_broadcasted = np.tile(qact4_scale, (outdata.shape[0], 1))
outdata = np.round(outdata / qact4_scale_broadcasted) + qact4_zp
#outdata = (outdata - qact2_zp) * qact2_scale_broadcasted

veri = load_act(qact4_file)
veri = veri.reshape(outdata.shape)
min_data = 1.0
error_count = compare_arrays(veri, outdata, min_data)
print("mlp qact total samples:%d, total error:%d" %(veri.size, error_count))

outdata = (outdata - qact4_zp) * qact4_scale_broadcasted

############################just repeat the above block# process 12 times
print("#####################")
###final output of block
block_file_head = "11"
qact4_scale_file  = "../export/block_qact4_b"+block_file_head+"_scale_cfg.txt"
qact4_zp_file     = "../export/block_qact4_b"+block_file_head+"_zeropoint_cfg.txt"
qact4_file        = "../export/block_qact4_b"+block_file_head+"_int8.txt"
qact4_scale = []
qact4_zp    = []
qact4_scale, qact4_zp = load_scale_and_zp(qact4_scale_file, qact4_zp_file)

veri = load_act(qact4_file)
veri = veri.reshape(outdata.shape)

outdata = (veri - qact4_zp) * qact4_scale
#############################################################################
### final  norm 
print("#########final norm")
fnorm_weight_file   = "../export/fnorm_weight.txt"
fnorm_bias_file     = "../export/fnorm_bias.txt"
fnorm_scalein_file  = "../export/fnorm_scalein.txt"
fnorm_scaleout_file = "../export/fnorm_scaleout.txt"

qact2_file       = "../export/qact2_int8.txt"
qact2_scale_file = "../export/qact2_scale_cfg.txt"
qact2_zp_file    = "../export/qact2_zeropoint_cfg.txt"
qact2_scale = []
qact2_zp = []

qact2_scale, qact2_zp = load_scale_and_zp(qact2_scale_file, qact2_zp_file)

def verify_layernorm2(layer, d_input, weight, bias, scalein, scaleout, d_scale, d_zp, out_d):
    d_wt = []
    d_bias   = []
    in_scale = []
    out_scale = []
    outdata = []
    x = d_input
    d_wt, d_bias = load_weight_and_bias(weight, bias)
    in_scale, out_scale =  load_scale_and_zp(scalein, scaleout)

    channel_nums = x.shape[-1]
    in_scale = in_scale.reshape(1, 1, -1)
    out_scale = out_scale.reshape(1, 1, -1)
    x_q = (x / in_scale).round()
    in_scale1 = in_scale.min()
    in_scale_mask = (in_scale / in_scale1).round()
    x_q = x_q * in_scale_mask
    # 计算均值和标准差
    mean_x_q = np.mean(x_q, axis=-1) * in_scale1
    std_x_q = (in_scale1 / channel_nums) * np.sqrt(channel_nums * np.sum(x_q**2, axis=-1) - np.sum(x_q, axis=-1)**2)
    # 计算矩阵乘积
    A = (in_scale1 / std_x_q).reshape(-1, 1) * d_wt.reshape(1, -1) / out_scale
    #print(A.shape)
    A_sign = np.sign(A)
    M, N = get_MN(np.abs(A))
    B = ((d_bias.reshape(1, -1) - (mean_x_q / std_x_q).reshape(-1, 1) * d_wt.reshape(1, -1)) / out_scale * 2**N).round()
    x_q = ((A_sign * M * x_q + B) / 2**N).round()
    x = x_q * out_scale
    x2 = x.reshape(d_input.shape)
    x2 = x2 / d_scale + d_zp
    print(layer, end='')
    print(" shape:", x2.shape)
    #output
    with open(out_d, 'r') as file:
        for line in file:
            line = line.strip()
            outdata.append(float(line))
    outdata = np.array(outdata)
    #outdata = outdata.reshape(x2.shape)
    min_data = 1.0
    error_count = compare_arrays(x2[0], outdata, min_data)
    print(layer, end='')
    print(" total samples:%d, total error:%d" %(x2[0].size, error_count))
    return outdata	


outdata = verify_layernorm2("final_norm", outdata, fnorm_weight_file, fnorm_bias_file, fnorm_scalein_file, fnorm_scaleout_file, qact2_scale, qact2_zp, qact2_file)

#############################################
print("#########final head")
outdata = (outdata - qact2_zp) * qact2_scale

#mlp_fc2
head_weight_file = "../export/head_wint.txt"
head_bias_file   = "../export/head_bias.txt"
head_scale_file  = "../export/head_scale_cfg.txt"
head_zp_file     = "../export/head_zeropoint_cfg.txt"

act_out_file       = "../export/act_out_int8.txt"
act_out_scale_file = "../export/act_out_scale_cfg.txt"
act_out_zp_file    = "../export/act_out_zeropoint_cfg.txt"

def verify_linear3(layer, d_input, wt, bias, scale, zp, pscale, pzp, outd, classnum  ):
	#[197,384]
	actlist = d_input
	d_pscale = []
	d_pzp    = []
	d_scale = []
	d_zp    = []
	d_wt    = []
	d_bias  = []
	outdata = []
	d_scale, d_zp = load_scale_and_zp(scale, zp)
	d_wt, d_bias = load_weight_and_bias(wt, bias)
	d_pscale, d_pzp = load_scale_and_zp(pscale, pzp)
	d_wt = d_wt.reshape(int(classnum), actlist.shape[0])
	for i in range(classnum):
	    d_wt[i] = (d_wt[i] - d_zp[i]) * d_scale[i]
	actlist = linear(actlist, d_wt, d_bias)
	actlist = actlist / d_pscale + d_pzp

	outdata = load_act(outd)
	outdata = outdata.reshape(actlist.shape)
	print(layer, end='')
	print(" shape:", outdata.shape)
	min_data = 1.0
	error_count = compare_arrays(actlist, outdata, min_data)
	print(layer, end='')
	print(" total samples:%d, total error:%d" %(actlist.size, error_count))
	return outdata, d_pscale, d_pzp
	
classnum = 1000	
outdata, mlpfc2_scale, mlpfc2_zp= verify_linear3("mlp_fc2", outdata, head_weight_file, head_bias_file, head_scale_file, head_zp_file, act_out_scale_file, act_out_zp_file, act_out_file, classnum)

#max_val = max(outdata)
max_idx = np.argmax(outdata)

print("The classification result is", max_idx)


    


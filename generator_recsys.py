from torch import nn
import torch
import torch.functional as F
import torch.nn.functional as F2
import time
import math
from torch.autograd import Variable
import numpy as np


class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        # self.conv1.weight = self.truncated_normal_(self.conv1.weight, 0, 0.02)
        # self.conv1.bias.data.zero_()

        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation*2)
        # self.conv1.weight = self.truncated_normal_(self.conv1.weight, 0, 0.02)
        # self.conv1.bias.data.zero_()

        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x): # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)
        out =  self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        out = F2.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation*2)
        out = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out = F2.relu(self.ln2(out))
        out = out + x
        return out

    def conv_pad(self, x, dilation):
        inputs_pad = x.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        inputs_pad = inputs_pad.unsqueeze(2)  # [batch_size, embed_size, 1, seq_len]
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        return inputs_pad

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class NextItNet_Decoder(nn.Module):

    def __init__(self, model_para):
        super(NextItNet_Decoder, self).__init__()
        self.model_para = model_para
        self.item_size = model_para['item_size']
        self.embed_size = model_para['dilated_channels']
        self.embeding = nn.Embedding(self.item_size, self.embed_size)
        stdv = np.sqrt(1. / self.item_size)
        self.embeding.weight.data.uniform_(-stdv, stdv) # important initializer
        # nn.init.uniform_(self.in_embed.weight, -1.0, 1.0)

        self.dilations = model_para['dilations']
        self.residual_channels = model_para['dilated_channels']
        self.kernel_size = model_para['kernel_size']
        rb = [ResidualBlock(self.residual_channels, self.residual_channels, kernel_size=self.kernel_size,
                            dilation=dilation) for dilation in self.dilations]
        self.residual_blocks = nn.Sequential(*rb)

        self.final_layer = nn.Linear(self.residual_channels, self.item_size)
        self.final_layer.weight.data.normal_(0.0, 0.01)  # initializer
        self.final_layer.bias.data.fill_(0.1)

    def forward(self, x, onecall=False): # inputs: [batch_size, seq_len]
        inputs = self.embeding(x) # [batch_size, seq_len, embed_size]

        dilate_outputs = self.residual_blocks(inputs)

        if onecall:
            hidden = dilate_outputs[:, -1, :].view(-1, self.residual_channels) # [batch_size, embed_size]
        else:
            hidden = dilate_outputs.view(-1, self.residual_channels) # [batch_size*seq_len, embed_size]
        out = self.final_layer(hidden)

        return out



# class NextItNet_Decoder(nn.Module):
#
#     def __init__(self, model_para):
#         super(NextItNet_Decoder, self).__init__()
#         self.model_para = model_para
#         self.item_size = model_para['item_size']
#         self.embed_size = model_para['dilated_channels']
#         self.embeding = nn.Embedding(self.item_size, self.embed_size)
#
#         self.dilations = model_para['dilations']
#         self.residual_channels = model_para['dilated_channels']
#         self.kernel_size = model_para['kernel_size']
#         residual_block = [nn.ModuleList([nn.Conv2d(self.residual_channels, self.residual_channels,
#                                                   kernel_size=(1, model_para['kernel_size']), padding=0, dilation=dilation),
#                                          nn.LayerNorm(self.residual_channels),
#                                          # Layer_norm(self.residual_channels),
#                                          # nn.ReLU(),
#                                          nn.Conv2d(self.residual_channels, self.residual_channels,
#                                                    kernel_size=(1, model_para['kernel_size']), padding=0, dilation=2*dilation),
#                                          nn.LayerNorm(self.residual_channels),
#                                          # Layer_norm(self.residual_channels),
#                                          # nn.ReLU()
#                                 ]) for dilation in self.dilations]
#         self.residual_blocks = nn.ModuleList(residual_block)
#
#         self.softmax_layer = nn.Linear(self.residual_channels, self.item_size)
#
#     def forward(self, x, onecall=False): # inputs: [batch_size, seq_len]
#         inputs = self.embeding(x) # [batch_size, seq_len, embed_size]
#
#         for i, block in enumerate(self.residual_blocks):
#             ori = inputs
#
#             inputs_pad = self.conv_pad(inputs, self.dilations[i])
#             # print(inputs_pad.size())
#             dilated_conv = block[0](inputs_pad).squeeze(2) # [batch_size, embed_size, seq_len]
#             dilated_conv = dilated_conv.permute(0, 2, 1)
#             relu1 = F2.relu(block[1](dilated_conv)) # [batch_size, seq_len, embed_size]
#
#             inputs_pad = self.conv_pad(relu1, self.dilations[i]*2)
#             # print(inputs_pad.size())
#             dilated_conv = block[2](inputs_pad).squeeze(2)  # [batch_size, embed_size, seq_len]
#             dilated_conv = dilated_conv.permute(0, 2, 1)
#             relu1 = F2.relu(block[3](dilated_conv))  # [batch_size, seq_len, embed_size]
#             inputs = ori + relu1
#
#         if onecall:
#             hidden = inputs[:, -1, :].view(-1, self.residual_channels) # [batch_size, embed_size]
#         else:
#             hidden = inputs.view(-1, self.residual_channels) # [batch_size*seq_len, embed_size]
#         out = self.softmax_layer(hidden)
#
#         return out
#
#     def conv_pad(self, inputs, dila_):
#         inputs_pad = inputs.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
#         inputs_pad = inputs_pad.unsqueeze(2)  # [batch_size, embed_size, 1, seq_len]
#         pad = nn.ZeroPad2d(((self.kernel_size - 1) * dila_, 0, 0, 0))
#         inputs_pad = pad(inputs_pad)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*self.dilations[i]]
#         return inputs_pad


class Layer_norm(nn.Module):
    def __init__(self, size):
        super(Layer_norm, self).__init__()
        # self.beta = torch.zeros(size, requires_grad=True)
        # self.gamma = torch.ones(size, requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(size))
        # nn.init.zeros_(self.beta)
        self.gamma = nn.Parameter(torch.ones(size))
        # nn.init.ones_(self.gamma)
        self.size = size
        self.epsilon = 1e-8

    def forward(self, x):
        shape = x.size()
        # print(shape)
        # print(x.mean(dim=2).size())
        # print(x.std(dim=2, unbiased=False).size())
        x = (x - x.mean(dim=2).view(shape[0], shape[1], 1)) / (x.std(dim=2, unbiased=False).view(shape[0], shape[1], 1) + self.epsilon)
        return self.gamma * x + self.beta


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F2.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class HGNN(nn.Module):
    def __init__(self, args, dim_capsule, dropout=0.5): #in_ch, n_class, n_hid,
        super(HGNN, self).__init__()
        self.dropout = dropout
        # self.hgc1 = HGNN_conv(in_ch, n_hid)
        # self.hgc2 = HGNN_conv(n_hid, n_class)
        self.hgc1 = HGNN_conv(dim_capsule * 2, dim_capsule)
        self.hgc2 = HGNN_conv(dim_capsule, 16)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)

        # 连接三种模态特征
        data_size = int(x.shape[0] / 3)  # 为了能够处理最后一轮batch_size不是确定值的情况
        embedding_t, embedding_a, embedding_v = torch.split(x, data_size, dim=0)
        logits_embeddings = torch.cat([embedding_t, embedding_a, embedding_v], dim=1)
        return logits_embeddings

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        nn.init.xavier_normal_(self.weight)
        self.bias = Parameter(torch.Tensor(out_ft))
        nn.init.normal_(self.bias)
        # if bias:
        #     self.bias = Parameter(torch.Tensor(out_ft))
        # else:
        #     self.register_parameter('bias', None)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.weight.data.uniform_(-stdv, stdv)
    #     if self.bias is not None:
    #         self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x
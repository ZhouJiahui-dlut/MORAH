import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
from torch.nn.init import xavier_normal

from src.CrossmodalTransformer import MULTModel
from src.StoG import *
from src.GraphCAGE import *
from src.HypergraphLearning import *
# from src.HGNN import *


class GCN_CAPS_Model(nn.Module):
    def __init__(self, args, label_dim, t_in, a_in, v_in, T_t, T_a, T_v,
                 MULT_d,
                 vertex_num,
                 dim_capsule,
                 routing,
                 dropout):
        super(GCN_CAPS_Model, self).__init__()
        self.d_c = dim_capsule # 胶囊维度
        self.n = vertex_num # 节点数
        self.T_t = T_t # 输入序列的长度
        self.T_a = T_a
        self.T_v = T_v
        self.dropout = dropout

        # encode part
        self.CrossmodalTransformer = MULTModel(args, t_in, a_in, v_in, MULT_d, dropout)
        # transformation from sequence to graph
        self.StoG = CapsuleSequenceToGraph(args, MULT_d, dim_capsule, vertex_num, routing, T_t, T_a, T_v)
        # Graph aggregate
        self.GraphAggregate = GraphCAGE(args, MULT_d, dim_capsule, vertex_num, routing, T_t, T_a, T_v)

        # 超图学习
        self.HyperG = HyperedgeConstruction(args, dim_capsule)
        # 超图卷积
        self.HyperGraphConv = HyperConv(args)
        # self.HyperGNN = HGNN(args, dim_capsule)

        # decode part
        self.fc1 = nn.Linear(in_features=3*dim_capsule*2, out_features=2*dim_capsule)
        self.fc2 = nn.Linear(in_features=2*dim_capsule, out_features=label_dim)
        # self.fc3 = nn.Linear(in_features=48, out_features=24)
        # self.fc4 = nn.Linear(in_features=24, out_features=label_dim)

    def forward(self, text, audio, video, batch_size):
        Z_T, Z_A, Z_V = self.CrossmodalTransformer(text, audio, video) # pre_Dimension(L,N,2MULT_d)
        text_vertex, audio_vertex, video_vertex, adj_t, adj_a, adj_v = self.StoG(Z_T, Z_A, Z_V, batch_size)
        nodes_t, nodes_a, nodes_v = self.GraphAggregate(text_vertex, audio_vertex, video_vertex, adj_t, adj_a, adj_v, batch_size)
        adjacency, nodes_list = self.HyperG(nodes_t, nodes_a, nodes_v, batch_size)
        logits = self.HyperGraphConv(adjacency, nodes_list)
        # logits = self.GraphAggregate(text_vertex, audio_vertex, video_vertex, adj_t, adj_a, adj_v, batch_size)

        # data_size = int(nodes_list.shape[0] / 3)
        # embedding_t, embedding_a, embedding_v = torch.split(nodes_list, data_size, dim=0)
        # logits = torch.cat([embedding_t, embedding_a, embedding_v], dim=1)

        output1 = torch.tanh(self.fc1(logits))
        output1 = F.dropout(output1, p=self.dropout, training=self.training)
        preds = self.fc2(output1) * 10
        return preds

        # 用HGNN进行图卷积
        # nodes_list, G = self.HyperG(nodes_t, nodes_a, nodes_v, batch_size)
        # logits = self.HyperGNN(nodes_list, G)
        # output1 = torch.tanh(self.fc3(logits))
        # preds = self.fc4(output1) * 10
        # return preds
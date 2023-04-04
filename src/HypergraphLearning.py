import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from scipy.sparse import coo_matrix


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

# def EuclideanDistances(a,b):
#     sq_a = a**2
#     sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
#     sq_b = b**2
#     sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
#     bt = b.t()
#     return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

# 输入稀疏矩阵的超图卷积
class HyperConv(nn.Module):
    def __init__(self, args):
        super(HyperConv, self).__init__()
        self.layers = args.layers

    def forward(self, adjacency, embedding):
        item_embeddings = embedding #初始节点嵌入
        item_embedding_layer0 = item_embeddings
        final = item_embedding_layer0.unsqueeze(0)
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)# 稀疏矩阵乘法
            final=torch.cat([final, item_embeddings.unsqueeze(0)], dim=0)
      #  final1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in final]))
      #  item_embeddings = torch.sum(final1, 0)
        item_embeddings = torch.sum(final, 0) / (self.layers+1) #最终的节点嵌入
        data_size = int(item_embeddings.shape[0]/3) # 为了能够处理最后一轮batch_size不是确定值的情况
        embedding_t, embedding_a, embedding_v = torch.split(item_embeddings, data_size, dim=0)
        logits_embeddings = torch.cat([embedding_t, embedding_a, embedding_v], dim=1)
        return logits_embeddings



class HyperedgeConstruction(nn.Module):
    def __init__(self, args, dim_capsule):
        super(HyperedgeConstruction, self).__init__()
        self.emb_size = dim_capsule * 2
        self.k_2 = args.k_2
        self.d_c = dim_capsule
        # self.W_dk = nn.Parameter(torch.Tensor(3))
        # nn.init.normal_(self.W_dk)
        # W_tk = W_k[0].repeat(batch_size)
        # W_ak = W_k[1].repeat(batch_size)
        # W_vk = W_k[2].repeat(batch_size)
        # self.W_dk = nn.Parameter(torch.diag(torch.cat([W_tk, W_ak, W_vk], 0)))
        # nn.init.xavier_normal(self.W_k)


    def forward(self, nodes_t, nodes_a, nodes_v, batch_size):
        # nodes_list = torch.cat([nodes_t, nodes_a, nodes_v], 0)
        nodes_num = len(nodes_t)*3
        # nodes_t_b = (nodes_t * self.W_dk[0])
        # nodes_a_b = (nodes_a * self.W_dk[1])
        # nodes_v_b = (nodes_v * self.W_dk[2])
        nodes_list = torch.cat([nodes_t, nodes_a, nodes_v], 0)
        H = torch.zeros([nodes_num, batch_size], dtype = torch.float32) #[N,M]
        # hyperedges = torch.zeros([batch_size, 3, self.emb_size], dtype=torch.float32)
        # hyperedges = set()
        for i in range(batch_size):
            # hyperedges.add(tuple({i, i+batch_size, i+batch_size*2}))
            # hyperedges[i][0] = nodes_t[i]
            # hyperedges[i][1] = nodes_a[i]
            # hyperedges[i][2] = nodes_v[i]
            H[i][i], H[i+batch_size][i], H[i+batch_size*2][i] = 1.0, 1.0, 1.0

        B = torch.sum(H, dim=0)
        B = torch.diag(B) # 边的度
        D = torch.sum(H, dim=1)
        D = torch.diag(D) # 点的度
        B_I = B.inverse() # 求逆矩阵
        H_T = H.transpose(1, 0)
        # 计算超边的表示，这里没考虑节点权重，超边权重都为1
        # hyperedges_list = torch.mm(torch.mm(B_I, H_T), nodes_list)

        # 计算超边的表示，加入节点权重矩阵(对角矩阵)
        # W_tk = self.W_k[0].repeat(batch_size)
        # W_ak = self.W_k[1].repeat(batch_size)
        # W_vk = self.W_k[2].repeat(batch_size)
        # W_kk = torch.diag(torch.cat([W_tk, W_ak, W_vk], 0))

        # 计算超边的表示，加入节点权重矩阵
        # hyperedges_list = torch.matmul(torch.matmul(B_I, H_T), torch.matmul(self.W_dk, nodes_list))

        # 计算超边的表示，加入节点权重矩阵
        # nodes_list = (torch.matmul(self.W_dk, nodes_list_b.permute(1,0,2))).reshape([-1,64])
        hyperedges_list = torch.mm(torch.mm(B_I, H_T), nodes_list)


        # #用内积计算扩展可能性，但由于节点特征相近无法区分
        # sigmoid1 = nn.Sigmoid()
        # hyperedges_list_T = hyperedges_list.transpose(0, 1)
        # expansion_possibility_test = torch.mm(hyperedges_list, hyperedges_list_T)  # [M,M]

        # 计算曼哈顿距离（L1范数）/p=2也可以求欧氏距离
        list_dist = torch.norm(hyperedges_list[:, None] - hyperedges_list, dim=2, p=1)
        # print("list_dist", list_dist)

        # 计算初始超边对应关联矩阵列的哈希值
        H_hash = []
        for i in range(batch_size):
            H_hash.append(hash(str(H[:, i])))

        # 1.为每条超边选择最可能扩展的k_2条超边
        k = list_dist.size()[0]
        if (k < self.k_2):
            self.k_2 = k
        list_dist_sort, idx_sort = torch.sort(list_dist, descending=True)  # descending为False，升序，为True，降序
        idx_max = idx_sort[:, :self.k_2]  # 截取排序后每行最大的k_2个元素的序号
        # 根据选出的k_2条超边建立新的超边，即更新关联矩阵
        for i in range(batch_size):
            h_new = H[:, i]
            for j in range(self.k_2):
                h_new = h_new + H[:, idx_max[i, j]]
            torch.clamp_max(h_new, max=1.0)
            h_new_hash = hash(str(h_new))
            if (h_new_hash in H_hash) is False:  # 判断是否为重复超边
                h_new = h_new.unsqueeze(1)
                H = torch.cat((H, h_new), dim=1)

        # # 消融实验：把已经被扩展的超边列清零
        # for i in range(batch_size):
        #     for j in range(self.k_2):
        #         H[:, idx_max[i, j]] = 0
        # # 将这些0列删除，只留下不为0的列
        # valid_cols = []
        # for col_idx in range(H.size(1)):
        #     if not torch.all(H[:, col_idx] == 0):
        #         valid_cols.append(col_idx)
        # H = H[:, valid_cols]

        # # 2.选择扩展可能性大于k_2的超边(确定阈值)
        # k = list_dist.size()[0]
        # mask = torch.zeros(k, k)
        # list_dist = torch.where(list_dist > self.k_2, list_dist, mask)
        # idx_value = torch.nonzero(list_dist)
        # # 想要去掉重复的，如[3, 4]和[4, 3]，没能实现
        # # idx_list = list()
        # # for i in range(idx_value.size()[0]):
        # #     idx_t = idx_value[i].tolist()
        # #     idx_list.append(idx_t)
        # # idx_value = torch.tensor(idx_list)
        # for i in range(idx_value.size()[0]):
        #     h_new = H[:, idx_value[i, 0]] + H[:, idx_value[i, 1]]
        #     h_new_hash = hash(str(h_new))
        #     if (h_new_hash in H_hash) is False:  # 判断是否为重复超边
        #          h_new = h_new.unsqueeze(1)
        #          H = torch.cat((H, h_new), dim=1)

        # 计算邻接矩阵（稀疏矩阵），为了便于调用超图卷积方法
        row, col = torch.nonzero(H, as_tuple=True)
        data = torch.ones(row.shape).detach().cpu()
        row = row.detach().cpu().numpy()
        col = col.detach().cpu().numpy()
        H_coo = coo_matrix((data, (row, col)), shape=H.shape, dtype=np.float32)
        H_T_coo = H_coo.T
        BH_T = H_T_coo.T.multiply(1.0 / H_T_coo.sum(axis=1).reshape(1, -1))  # [N,M]
        BH_T = BH_T.T  # [M,N]
        DH = H_coo.T.multiply(1.0 / H_coo.sum(axis=1).reshape(1, -1))  # [M,N]
        DH = DH.T  # [N,M]
        DHBH_T = np.dot(DH, BH_T)  # [N,N]

        adjacency = DHBH_T.tocoo()
        # 将邻接矩阵从coo_matrix转化为tensor
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = trans_to_cuda(torch.sparse.FloatTensor(i, v, torch.Size(shape)))
        return adjacency, nodes_list

        # G = _generate_G_from_H(H)
        # return nodes_list, G

# def _generate_G_from_H(H, variable_weight=False):
#     """
#     calculate G from hypgraph incidence matrix H
#     :param H: hypergraph incidence matrix H
#     :param variable_weight: whether the weight of hyperedge is variable
#     :return: G
#     """
#     H = H.cpu().numpy()
#     H = np.array(H)
#     n_edge = H.shape[1]
#     # the weight of the hyperedge
#     W = np.ones(n_edge)
#     # the degree of the node
#     DV = np.sum(H * W, axis=1)
#     # the degree of the hyperedge
#     DE = np.sum(H, axis=0)
#
#     invDE = np.mat(np.diag(np.power(DE, -1)))
#     DV2 = np.mat(np.diag(np.power(DV, -0.5)))
#     W = np.mat(np.diag(W))
#     H = np.mat(H)
#     HT = H.T
#
#     if variable_weight:
#         DV2_H = DV2 * H
#         invDE_HT_DV2 = invDE * HT * DV2
#         return DV2_H, W, invDE_HT_DV2
#     else:
#         G = DV2 * H * W * invDE * HT * DV2
#         G = torch.from_numpy(G).cuda()
#         # G = torch.tensor(G, dtype=torch.float32)
#         return G.float()

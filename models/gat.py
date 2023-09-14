import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # print(input.shape)
        # print(self.W.shape)
        h = torch.matmul(input, self.W)  # shape [N, out_features]
        # print(h.shape)
        # N = h.size()[0]
        N = h.size()[1]
        batch_len = h.size()[0]

        # 生成N*N的嵌入,repeat两次，两种方式，这个地方应该是为了把不同的sample拼接。为下一步求不同样本交互做准备。
        # a_input是由Whi和Whjconcat得到，对应论文里的Whi | | Whj
        a_input = torch.cat([h.repeat(1, 1, N).view(batch_len,N * N, -1), h.repeat(1,N, 1)], dim=1).view(batch_len,N, -1,
                                                                                          2 * self.out_features) # shape[N, N, 2*out_features]
        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1,
        #                                                                                   2 * self.out_features)  # shape[N, N, 2*out_features]
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N,N,1] -> [N,N]

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15 * torch.ones_like(e)  # 维度大小与e相同，所有元素都是-9*10的15次方
        attention = torch.where(adj > 0, e, zero_vec)
        print(attention.shape)
        # 当adj > 0，即两结点有边，则用gat构建的矩阵e，若adj = 0, 则另其为一个很大的负数，这么做的原因是进行softmax时，这些数就会接近于0了。
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [N,N], [N, out_features] --> [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):#batch_len表示一个batch里句子个数
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=-1)


# text_out = torch.rand(2,4,600)
#
# adj = torch.tensor([[[1, 0, 1,0], [0, 0, 0,0], [1, 1, 0,1],[1, 1, 0,1]],
#                     [[0, 0, 0,0], [0, 1, 0,0], [1, 0, 0,1],[0, 1, 0,1]]])
# print(text_out.shape, adj.shape)#第一个维度必须相同
# layer1 = GAT(nfeat=600, #必须和text_out第二维度一样
#             nhid=600,
#             nclass=600,#最后的维度
#             dropout=0.6,
#             nheads=3,
#             alpha=0.2)
# layer2 = GAT(nfeat=600, #必须和text_out第二维度一样
#             nhid=600,
#             nclass=600,#最后的维度
#             dropout=0.6,
#             nheads=2,
#             alpha=0.2)
# a = layer1(text_out, adj)#3*300
#
# print("a")
# print(a.shape)
# b = layer2(a,adj)
# print(b.shape)
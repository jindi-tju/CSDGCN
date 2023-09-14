import torch
import torch.nn as nn
import torch.nn.functional as F

class SAGELayer(nn.Module):

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(SAGELayer, self).__init__()
        self.weight = torch.Tensor(in_features*2, out_features)  # 128*128
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, batch_norm=True):  # adj1 [l,l] input 就是上一层的x ([32bs , l, 128])
        # input mes 拼接  再乘w  再relu dropout
        # 公式里写的是 concat，这里是 element-wise 的和。

        mes = torch.matmul(adj, input)  # l*l  * l*128  == l*128
        # con = input + mes  # 按元素相加
        # output = torch.matmul(con, self.weight)
        con1 = torch.cat((input, mes), dim=-1)   # 最后一维拼接 l* 256
        output = torch.matmul(con1, self.weight)  #  self.weight 256*128



        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class GraphSAGE(nn.Module):  #  改调用那里
    def __init__(self, nfeat, nhid, nclass, graph_hops, dropout, batch_norm=False):
        super(GraphSAGE, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(SAGELayer(nfeat, nhid, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(SAGELayer(nhid, nhid, batch_norm=batch_norm))

        self.graph_encoders.append(SAGELayer(nhid, nclass, batch_norm=False))


    def forward(self, x, node_anchor_adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, node_anchor_adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, node_anchor_adj)
        return x

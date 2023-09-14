# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from layers.bert import get_bert_output
from models.inconsistency import Scaled_Dot_Product_Attention_pos, Scaled_Dot_Product_Attention_neg

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class CSDGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(CSDGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc3 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc4 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc5 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc6 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.fc = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)
        self.fc1 = nn.Linear(600, 300)
        self.text_embed_dropout = nn.Dropout(0.5)
        self.kg_pos = Scaled_Dot_Product_Attention_pos()
        self.kg_neg = Scaled_Dot_Product_Attention_neg()

    def forward(self, inputs, dataset_type):
        datas, text_indices, adj, sentic_adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)

        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)  # bert和lstm都试一下哪个效果好
        augu_list, augu_len, augu_index = get_bert_output(datas, dataset_type)

        # 最长（句子长度+增广词数）
        max = 0
        for m in range(len(augu_len)):
            temp = text_len[m] + augu_len[m]
            if temp > max:
                max = temp
        # 补0
        for p in range(len(text_out)):
            text_out1 = torch.zeros(len(text_out), max, 600)
            a1 = torch.zeros(max - len(text_out[p]), 600)
            text_out1[p] = torch.cat([text_out[p], a1], dim=0)

        # 情感图属性加进去(commonsense augment)
        for i in range(len(augu_list)):
            position = text_len[i]
            if position != max:  # 如果句子长度最长，加上0个常识仍然为最长，position处可能会越界，处理一下
                for j in range(augu_len[i]):
                    text_out1[i][position] = torch.tensor(augu_list[i][j][0])
                    position = position + 1
        text_out1 = text_out1.to(device)

        # 依赖图替换属性(commomsense replacement)
        for i, au_index in enumerate(augu_index):
            if au_index != []:
                for j, index in enumerate(au_index):
                    text_out[i][index] = torch.tensor(augu_list[i][j][0])

        # 情感图
        x = F.relu(self.gc2(text_out1, sentic_adj))
        x = F.relu(self.gc4(x, sentic_adj))
        x = F.relu(self.gc6(x, sentic_adj))

        # 依赖图
        y = F.relu(self.gc1(text_out, adj))
        y = F.relu(self.gc3(y, adj))
        y = F.relu(self.gc5(y, adj))

        alpha_mat = torch.matmul(y, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        y = torch.matmul(alpha, text_out).squeeze(1).unsqueeze(0)
        alpha_mat1 = torch.matmul(x, text_out1.transpose(1, 2))
        alpha1 = F.softmax(alpha_mat1.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha1, text_out1).squeeze(1).unsqueeze(0)  # 1 batsize 600
        temp = torch.cat([x, y], dim=0)
        temp = torch.mean(temp, dim=0, keepdim=True).squeeze(0)
        output = self.fc(temp)
        return output

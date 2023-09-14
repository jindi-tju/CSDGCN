import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Scaled_Dot_Product_Attention_pos(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention_pos, self).__init__()

    def forward(self, Q, K, V, scale,kg_sim):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1).to(device))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        beta = torch.mul(attention, kg_sim.to(device))
        beta = F.softmax(beta,dim = -1)
        # print('beta size:',beta.size()) #128,1,5
        # print('v size:',V.size())#128,5,80
        context = torch.matmul(beta, V.to(device))
        # print('v after attention:',context.size()) #128,1,80
        return context

class Scaled_Dot_Product_Attention_neg(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention_neg, self).__init__()

    def forward(self, Q, K, V, scale, kg_sim):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = -1*torch.matmul(Q, K.permute(0, 2, 1).to(device))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = -1*F.softmax(attention, dim=-1)
        beta = torch.mul(attention, kg_sim.to(device))
        beta = F.softmax(beta, dim=-1)
        # print('beta size:', beta.size())  # 128,1,5
        # print('v size:',V.size())#128,5,80
        context = torch.matmul(beta, V.to(device))
        # print('v after attention:', context.size())  # 128,1,80
        # context = torch.matmul(attention, V)
        return context

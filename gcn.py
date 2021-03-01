from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj.to_dense(), support)
        # print(adj.to_dense()[0, :6, :6])
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        self.max_len = 133
        self.nclass = nclass
        '''
        self.ratio = nn.Parameter(torch.FloatTensor(1))
        self.bias = nn.Parameter(torch.FloatTensor(1))
        '''
        # self.embedding2 = nn.Linear(7, nfeat//2, bias=True)
        # self.embedding1 = nn.Linear(53, nfeat//2, bias=True)
        self.embedding = nn.Linear(nfeat, nhid, bias=True)
        self.gc1 = GraphConvolution(nhid, nhid*2)
        # self.bn1 = nn.BatchNorm1d(2*nhid)
        self.gc2 = GraphConvolution(nhid*2, nhid)
        # self.bn2 = nn.BatchNorm1d(nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        # self.bn3 = nn.BatchNorm1d(133)
        # self.gc4 = GraphConvolution(nhid//2, nclass)
        # self.bn = nn.BatchNorm1d(nclass)
        # self.gc5 = GraphConvolution(nfeat, 1)
        # self.dropout = dropout
        '''
        self.pool = nn.AdaptiveMaxPool1d(7)
        self.final_layer = nn.Linear(32, 1, bias=True)
        '''
        # self.fc1 = nn.Linear(self.max_len-1, 7, bias=True)
        # self.final_layer = nn.Linear(nclass*8, 1, bias=True)
        
        self.fc1 = nn.Linear(in_features=132*nclass,out_features=14)
        self.fc2 = nn.Linear(in_features=16,out_features=1)

    def forward(self, x, adj):
        '''
        x1 = x[:, :, :53]
        x2 = x[:, :, 53:]
        x1 = x1 * self.ratio + self.bias
        x = torch.cat((x1, x2), dim=2)
        '''
        # x2 = self.embedding2(x2)
        # x = F.relu(self.embedding(x))
        # x = F.dropout(x, 0.2, training=self.training)
        x = self.embedding(x)
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, 0.1, training=self.training)
        # x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.graph_pooling(x, adj)
        # print(x)
        # x = F.dropout(x, 0.3, training=self.training)
        x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, 0.1, training=self.training)
        # x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.graph_pooling(x, adj)
        x = F.relu(self.gc3(x, adj))
        # x = self.graph_pooling(x, adj)
        # x = F.dropout(x, 0.1, training=self.training)
        # x = self.bn3(x)
        # x = F.relu(self.gc4(x, adj))
        # x = self.graph_pooling(x, adj)
        # x = F.relu(self.gc5(x, adj))
        '''
        x = x.transpose(1, 2) # b * nclass * atom_num
        
        # x = self.bn(x)
        
        fst_x = x[:, :, :1]
        x = x[:, :, 1:]
        x = F.dropout(x, 0.2, training=self.training)
        x = F.relu(self.fc1(x))
        x = torch.cat((fst_x, x), dim=2).reshape(x.shape[0], 1, -1)
        # x = F.dropout(x, 0.1, training=self.training)
        # x = self.bn(x)
        x = self.final_layer(x).sigmoid()
        '''
        fst_x = x[:, :1].reshape(x.shape[0], -1)
        x = x[:, 1:]
        x = x.view(x.size(0), -1)
        x = F.dropout(x, 0.3, training=self.training)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, 0.3, training=self.training)
        # x = self.bn(x)
        x = torch.cat((fst_x, x), dim=1)
        x = self.fc2(x).sigmoid()
        return x
    
    def graph_pooling(self, x, adj):
        dense_adj = adj.to_dense()
        x = torch.matmul(dense_adj, x)
        
        return x
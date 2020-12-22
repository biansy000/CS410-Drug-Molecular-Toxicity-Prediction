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
    def __init__(self, nfeat, nhid, nclass, dropout=0.1):
        super(GCN, self).__init__()
        self.max_len = 133

        self.embedding = nn.Linear(nfeat, 2*nfeat, bias=True)
        self.gc1 = GraphConvolution(2*nfeat, 2*nhid)
        # self.bn1 = nn.BatchNorm1d(2*nhid)
        self.gc2 = GraphConvolution(2*nhid, nhid)
        # self.bn2 = nn.BatchNorm1d(nhid)
        self.gc3 = GraphConvolution(nhid, nfeat)
        # self.bn3 = nn.BatchNorm1d(133)
        self.gc4 = GraphConvolution(nfeat, 1)
        # self.gc5 = GraphConvolution(nfeat, 1)
        self.dropout = dropout
        # self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(self.max_len-1, 3, bias=True)
        self.final_layer = nn.Linear(4, 1, bias=True)
        # self.fc2 = nn.Linear(self.max_len//8, 1, bias=True)

        # self.bn_parameter1 = nn.Parameter(torch.FloatTensor(nhid))
        # self.bn_parameter2 = nn.Parameter(torch.FloatTensor(nhid))
    '''
    def _initialize(self):
        self.bn_parameter1 = 1
        self.bn_parameter2 = 0
    '''
    def forward(self, x, adj):
        neibs = adj.to_dense() > 1e-5
        x = F.relu(self.embedding(x))
        x = F.relu(self.gc1(x, adj))
        x = self.graph_pooling(x, neibs)
        # x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.graph_pooling(x, neibs)
        # x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(self.gc3(x, adj))
        x = self.graph_pooling(x, neibs)
        # x = self.bn3(x)
        x = F.relu(self.gc4(x, adj))
        # x = F.relu(self.gc5(x, adj))
        # x = x[:, 0].sigmoid()
        x = x.reshape(x.shape[0], 1, -1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.pool(x).sigmoid()
        # x = F.relu(self.fc1(x))

        fst_x = x[:, :, :1]
        x = x[:, :, 1:]
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc1(x)
        x = torch.cat((fst_x, x), dim=2)
        x = self.final_layer(x).sigmoid()
       
        return x
    
    def graph_pooling(self, x, neibs):
        new_x = torch.zeros_like(x)
        num_nodes = x.shape[1]
        for b in range(x.shape[0]):
            for i in range(num_nodes):
                mask = neibs[b, i]
                if (mask == False).all():
                    continue
                neib_feat = x[b, mask, :].reshape(-1, x.shape[2]) # neighbour * feat
                max_feat = torch.max(neib_feat, dim=0).values
                # print(torch.max(torch.cat((neib_feat, x[b, i:i+1]), dim=0), dim=0).values.shape)
                new_x[b, i] = max_feat
        
        return new_x
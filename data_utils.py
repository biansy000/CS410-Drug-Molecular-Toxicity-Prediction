import numpy as np
import copy
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from gcn_utils import read_data, get_adj_feat

paths = {'test': "../code/data/test",
        'train': "../code/data/train",
        'valid': "../code/data/validation"}

tmp_zeros = [0. for i in range(53)]
tmp_ones = [1. for i in range(53)]
mean_feature = np.array(tmp_zeros+[2.8033829e-01, 3.5248524e-01, 5.5676805e-06, 5.6651151e-03,
                            1.2676866e-01, 3.9901712e-05, 4.4107165e-02])
mean_std = np.array(tmp_ones+[0.7668853,  0.9752085,  0.05274452, 0.07659069, 0.48673064,
                        0.01005699, 0.20410864])


class Smiles(Data.Dataset):
    def __init__(self, data_choice, use_edgeweight=True, dtype=torch.float32, pos_weight=7.0, device=torch.device('cpu') ):
        # data augment: generate different SMILES for one chemical
        super(Smiles, self).__init__()
        self.data_choice = data_choice
        rdata = read_data(self.data_choice, pos_weight=pos_weight)
        self.data = get_adj_feat(rdata, self.data_choice)
        self.use_edgeweight = use_edgeweight
        
        # normalize([features for features in self.data])

        self.dtype = dtype
        self.device = device
                
    def __getitem__(self, idx):
        data = self.data[idx]
        if self.use_edgeweight:
            adj = data['edge_weight']
        else:
            adj = data['adj']
        # print(adj)
        values = adj.data
        indices = np.vstack((adj.row, adj.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj.shape
        # print(shape)

        features =  data['features']
        features = (features - mean_feature)/mean_std
        if self.data_choice == 'test':
            return data['name'], \
                torch.sparse.FloatTensor(i, v, torch.Size(shape)).to('cuda'), \
                torch.tensor(features, device=self.device, dtype=self.dtype)
            
        # print(features.shape)
        label = data['label']
        assert label == 0 or label == 1, f'{label}'
        weight = data['weight']

        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to('cuda'), \
                torch.tensor(features, device=self.device, dtype=self.dtype), \
                torch.tensor(label, device=self.device, dtype=self.dtype), \
                torch.tensor(weight, device=self.device, dtype=self.dtype)

    def __len__(self):
        return len(self.data)

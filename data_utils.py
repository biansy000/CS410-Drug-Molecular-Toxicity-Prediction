import numpy as np
import copy
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from gcn_utils import read_data, get_adj_feat

paths = {'test': "./data/test",
        'train': "./data/train",
        'valid': "./data/validation"}

tmp_zeros = [0. for i in range(27)]
tmp_ones = [0.2 for i in range(27)]
'''
mean_feature = np.array(tmp_zeros+[2.8033829e-01, 3.5248524e-01, 5.5676805e-06, 5.6651151e-03,
                            1.2676866e-01, 3.9901712e-05, 4.4107165e-02])
mean_std = np.array(tmp_ones+[0.7668853,  0.9752085,  0.05274452, 0.07659069, 0.48673064,
                        0.01005699, 0.20410864])
'''
mean_feature = np.array([9.8081432e-02, 9.3598152e-03, 2.1026660e-02, 4.2180414e-04, 2.7583044e-03,
 3.7299274e-04, 1.7765507e-03, 2.6984414e-04, 2.3945213e-05, 1.4459225e-03,
 2.5787152e-05, 1.4735516e-05, 3.5917819e-05, 1.8511491e-04, 9.9464734e-05,
 3.9601699e-05, 8.2887273e-06, 7.3677579e-06, 8.2887273e-06, 1.0130667e-05,
 6.4467881e-06, 5.5258183e-06, 4.6048485e-06, 5.5258183e-06, 4.6048485e-06,
 4.6048485e-06, 4.4206547e-05, 2.7823049e-01, 3.4983495e-01, 5.5258183e-06,
 5.6225201e-03, 1.2581553e-01, 3.9601699e-05, 4.3775532e-02])
mean_std = np.array([0.29554033, 0.09595011, 0.14256993, 0.0205292,  0.05237684, 0.01930591,
 0.04207614, 0.01642252, 0.00489327, 0.03797172, 0.00507797, 0.00383863,
 0.00599293, 0.0136032,  0.00997222, 0.00629274, 0.00287899, 0.00271434,
 0.00287899, 0.00318284, 0.00253904, 0.00235069, 0.00214588, 0.0023507,
 0.00214588, 0.00214588, 0.00664851, 0.75785226, 0.9730329,  0.05254586,
 0.0763038,  0.48522747, 0.01001912, 0.20337662])

class Smiles(Data.Dataset):
    def __init__(self, data_choice, use_edgeweight=True, dtype=torch.float32, pos_weight=7.0, device=torch.device('cpu') ):
        # data augment: generate different SMILES for one chemical
        super(Smiles, self).__init__()
        self.data_choice = data_choice
        rdata = read_data(self.data_choice, pos_weight=pos_weight)
        self.data = get_adj_feat(rdata, self.data_choice)
        self.use_edgeweight = use_edgeweight

        for item in self.data:
            item['features'] = (item['features'] - mean_feature)/mean_std
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

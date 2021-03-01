import torch
import numpy as np
import data_utils
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence
from data_utils import Smiles, read_data
from gcn import GCN
from torch.utils.data import DataLoader
from metrics import EvaMetric, AUCMetric
from criterion import weighted_ce_loss, weighted_bce_loss, weighted_focal_loss
from all_cfg import all_cfg
from utils import sigmoid
from tqdm import tqdm

my_cfg = all_cfg['trivial_res']

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def write_test(model_name):
    test_dataset = Smiles(data_choice='test', pos_weight=my_cfg['pos_weight'], device=device)
    test_loader = DataLoader(test_dataset, batch_size=my_cfg['batch_size'], shuffle=False)

    best_model = GCN(34, 32, 2)
    best_model.load_state_dict(torch.load(model_name))
    if torch.cuda.is_available():
        best_model = best_model.cuda()
        
    best_model.eval()
    print('\nStarting test ...')
    results = []
    for i, (names, adj, features) in enumerate(test_loader):
        res = best_model(features, adj).detach().cpu()
        res = res.reshape(-1)
        for name, my_res in zip(names, res):
            results.append({'name': name, 'res': my_res})

    exp_num = my_cfg['exp_num']
    model_name = model_name.split('.')[-2][1:]
    with open(f'./data/test/output_{model_name}.txt', "w") as f:
        f.write('Chemical,Label\n')
        assert len(results) == 610
        for i in range(len(results)):
            my_name = results[i]['name']
            my_res = results[i]['res']
            my_res = my_res.detach().cpu().numpy()
            f.write(f'{my_name},{my_res}\n')
    return 



if __name__ == "__main__":
    write_test('./weights/best_model.pth')
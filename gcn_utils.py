import numpy as np
from numpy import loadtxt
import os
import copy
import json
import pandas as pd
import networkx as nx
from pysmiles import read_smiles, write_smiles, fill_valence
from networkx.linalg.graphmatrix import adjacency_matrix
import rdkit.Chem as Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from scipy import sparse

paths = {'test': "../code/data/test",
        'train': "../code/data/train",
        'valid': "../code/data/validation"}

element_dict = {6: 0, 7: 1, 8: 2, 35: 3, 17: 4, 11: 5, 16: 6, 15: 7, 20: 8, 9: 9, 5: 10, 
                33: 11, 13: 12, 53: 13, 14: 14, 19: 15, 24: 16, 30: 17, 34: 18, 40: 19, 26: 20, 
                50: 21, 60: 22, 29: 23, 79: 24, 82: 25, 81: 26, 51: 27, 48: 28, 46: 29, 22: 30, 
                1: 31, 78: 32, 49: 33, 56: 34, 47: 35, 66: 36, 80: 37, 3: 38, 70: 39, 25: 40, 
                12: 41, 27: 42, 28: 43, 4: 44, 32: 45, 83: 46, 23: 47, 38: 48, 42: 49, 44: 50, 63: 51, 21: 52}


def read_data(choice='train', pos_weight=7.0):
    if choice == 'test':
        return read_test_data()
    path = paths[choice]
    rdata = []

    df_smiles = pd.read_csv(os.path.join(path, 'names_smiles.txt'))
    list_smiles = df_smiles.values.astype(str).tolist()
    df_labels = pd.read_csv(os.path.join(path, 'names_labels.txt'))
    list_labels = df_labels.values.astype(str).tolist()

    for i in range(len(list_labels)):
        weight = 1.0
        if int(list_labels[i][1]) > 0.5:
            weight = pos_weight
        tmp = {'name': list_smiles[i][0], 'SMILES': list_smiles[i][1], \
            'label': int(list_labels[i][1]), 'weight': weight}
        rdata.append(tmp)

    return rdata


def read_test_data():
    path = paths['test']
    rdata = []

    df_smiles = pd.read_csv(os.path.join(path, 'names_smiles.txt'))
    list_smiles = df_smiles.values.astype(str).tolist()

    for i in range(len(list_smiles)):
        tmp = {'name': list_smiles[i][0], 'SMILES': list_smiles[i][1]}
        rdata.append(tmp)

    return rdata

'''
mol = Chem.MolFromSmiles("CC(CC)C")
main_adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
# print(main_adjacency_matrix)
print(mol.GetAtomWithIdx(1).GetAtomicNum())
'''

def get_adj_feat(rdata, choice='train'):
    new_data = []
    wrong_cnt = 0
    max_len = 133
    for i, data in enumerate(rdata):
        try:
        # if True:
            smiles = data['SMILES']
            if choice == 'test' and (i == 303 or i == 344):
                smiles = smiles.replace('c', 'C')
                smiles = smiles.replace('n', 'N')
            mol = Chem.MolFromSmiles(smiles)
            adj = GetAdjacencyMatrix(mol)
            padded_adj = np.zeros((max_len, max_len), dtype=np.int16)
            
            padded_adj[1:adj.shape[0]+1, 1:adj.shape[1]+1] = adj
            # add a dummy node
            padded_adj[0, 1:adj.shape[1]+1] = 1
            # padded_adj[:adj.shape[0]+1, 0] = 1
            
            # padded_adj[:adj.shape[0], :adj.shape[1]] = adj

            unit = np.eye(max_len)
            A_hat = padded_adj + unit
            diagonal = np.zeros_like(padded_adj)
            for k in range(0, max_len):
                diagonal[k, k] = np.sum(A_hat[k])
            
            # mask = np.array([[i, i] for i in range(max_len)], dtype=np.int32)
            sqrt_diag_inv = np.zeros_like(diagonal, dtype=np.float32)
            # sqrt_diag = np.sqrt(diagonal)
            for k in range(max_len):
                sqrt_diag_inv[k, k] = 1/np.sqrt(diagonal[k, k])
            edge_weight = np.matmul(sqrt_diag_inv, A_hat)
            edge_weight = np.matmul(edge_weight, sqrt_diag_inv)
            
            for k in range(adj.shape[0]+1, max_len):
                edge_weight[k, k] = 0

            my_features = np.zeros((max_len, 60), dtype=np.float32)
            for i, atom in enumerate(mol.GetAtoms()):
                my_features[i+1] = get_feature(atom)

            data['adj'] = sparse.coo_matrix(padded_adj)
            data['features'] = np.array(my_features)
            data['edge_weight'] = sparse.coo_matrix(edge_weight)
            new_data.append(data)
        except:
            print(i, rdata[i]['SMILES'])
            wrong_cnt += 1
    
    print('Discarded atoms', wrong_cnt)
    print('max len', max_len)
    '''
    print('length of a dict', elements)
    with open("elements.txt", "a") as f:
        for key, item in elements.items():
            f.write(f'{key}, ')
        f.write('\n')
    '''
    return new_data

'''
rdata = read_train_data()
get_adjacent_m(rdata)
'''

def get_feature(atom):
    degree = atom.GetDegree()
    atom_num = atom.GetAtomicNum()
    valence = atom.GetExplicitValence()
    imp_valence = atom.GetImplicitValence()
    formal_charge = atom.GetFormalCharge()
    exp_hs = atom.GetNumExplicitHs()
    # imp_hs = atom.GetNumImplicitHs()
    electron = atom.GetNumRadicalElectrons()
    aromatic = atom.GetIsAromatic()
    idx = atom.GetIdx()

    onehot = [0. for i in range(53)]
    onehot[element_dict[int(atom_num)]] = 1.

    other_feature = [degree, valence, formal_charge, exp_hs,
                imp_valence, electron, aromatic]
    feature = np.array(onehot + other_feature, dtype=np.float32)
    '''
    feature = [degree, atom_num, valence, formal_charge, exp_hs,
                imp_hs, electron, aromatic]
    '''
    return feature

if __name__ == "__main__":
    rdata = read_data('train', pos_weight=1)
    data = get_adj_feat(rdata, 'test')
    
    features = np.array([item['features'] for item in data])
    print(features.shape)
    features = features.reshape(-1, features.shape[-1])
    print(features.mean(axis=0))
    print(np.std(features, axis=0))
    

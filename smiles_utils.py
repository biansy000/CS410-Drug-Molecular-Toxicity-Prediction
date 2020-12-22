import numpy as np
from numpy import loadtxt
import os
import copy
import json
import pandas as pd
import networkx as nx
from pysmiles import read_smiles, write_smiles, fill_valence

paths = {'test': "./data/test",
        'train': "./data/train",
        'valid': "./data/validation"}

def split_smiles(smiles, element_list):
    special_case = ['b', 'c', 'o', 'p', 's']
    idx_list = []
    for i, ele in enumerate(smiles):
        ele = str(ele)
        #assert ele != 'n', 'SIMPLIFICATION FAILS'
            
        if ele.islower() and (not ele in special_case) and i > 0 and\
                str(smiles[i-1]).isupper(): # is the suffix of an element
            continue

        if ele.isupper() and i < len(smiles) - 1 and str(smiles[i+1]).islower() \
                and (not str(smiles[i+1]) in special_case): # an element with 2 chars
            ele = ele + str(smiles[i+1])

        idx = element_list.index(ele)
        idx_list.append(idx)
    
    return idx_list


def find_elements():
    # list all the elements appeared
    special_case = ['b', 'c', 'o', 'p', 's']
    element_list = []
    longest_len =0
    for path_name in paths:
        path = paths[path_name]
        df_smiles = pd.read_csv(os.path.join(path, 'names_smiles.txt'))
        smiles_list = np.array(df_smiles.iloc[:, 1])

        for smiles in smiles_list:
            mol = read_smiles(smiles)
            for node in mol.nodes:
                if 'stereo' in mol.nodes[node]:
                    mol.nodes[node].pop('stereo') # discard stereo infomation by hand

            new_smiles = write_smiles(mol)
            length = 0
            for i, ele in enumerate(new_smiles):
                ele = str(ele)
                #assert ele != 'n', 'SIMPLIFICATION FAILS'
                    
                if ele.islower() and (not ele in special_case) and i > 0 and\
                        str(new_smiles[i-1]).isupper(): # is the suffix of an element
                    continue
                if ele.isupper() and i < len(new_smiles) - 1 and str(new_smiles[i+1]).islower() \
                        and (not str(new_smiles[i+1]) in special_case): # an element with 2 chars
                    ele = ele + str(new_smiles[i+1])

                length += 1
                if not ele in element_list:
                    element_list.append(ele)
                
                if length > longest_len:
                    longest_len = length

    print(element_list)
    with open('element_list.txt', 'w') as f:
        for item in element_list:
            f.write("%s " % item)
        f.write(f'{longest_len}')


def write_dict(name, onehots, smiles, label=None, weight=None, aug_times=None):
    if label != None:
        return {'name': name, 'onehots': onehots, 'SMILES': smiles, \
                'label': label, 'weight': weight, 'aug_times': aug_times}
    else:
        return {'name': name, 'onehots': onehots, 'SMILES': smiles}


def generate_onehots(smiles_list, element_list):
    onehots_list = []
    for smiles in smiles_list:
        smiles_tmp = split_smiles(smiles, element_list)
        onehots = np.zeros((len(element_list), len(smiles_tmp)) )
        for i in range(len(smiles_tmp)):
            onehots[smiles_tmp[i], i] = 1

        onehots_list.append(onehots)
    return onehots_list



def augment_smiles(rdata, data_choice, make_1d_pading=True):
    if os.path.exists(f'data/{data_choice}/data.json'):
        with open(f'data/{data_choice}/data.json', 'r') as f:
            new_data = json.load(f)

        return new_data, 320

    if not os.path.exists('./element_list.txt'):
        print('no element list found!\n')
        find_elements() # find all appeared elements if the list has not been abtained
    
    new_data = []
    with open('element_list.txt') as f:
        lines = f.readlines()
        element_list = lines[0].split(sep=' ')[:-1]
        #longest_len = int(lines[0].split(sep=' ')[-1])
        longest_len = 320

    for item in rdata:
        name = item['name']
        smiles = item['SMILES']
        if data_choice != 'test':
            label = item['label']
            weight = item['weight']

        mol = read_smiles(smiles)
        for node in mol.nodes:
            if 'stereo' in mol.nodes[node]:
                mol.nodes[node].pop('stereo') # discard stereo infomation by hand
        
        degrees = np.array([mol.degree(idx) for idx in mol.nodes])
        # find leaf nodes to generate different smiles for one mol
        leaf_nodes = np.array(list(mol.nodes), dtype=int)[degrees == 1]
        if leaf_nodes.shape[0] == 0:
            leaf_nodes = [0]

        try:
            if len(leaf_nodes) > 5 and data_choice != 'test':
                length = len(leaf_nodes)
                idx_list = [idx for idx in range(0, length, length//4)]
                leaf_nodes = leaf_nodes[idx_list] # at most 5 examples

            new_smiles_list = [write_smiles(mol, start=list(mol.nodes)[leaf_node]) for leaf_node in leaf_nodes]
            onehots_list = generate_onehots(new_smiles_list, element_list)
            if data_choice != 'test':
                tmp = [write_dict(name, onehots, smiles, label, weight, len(leaf_nodes)) \
                        for onehots, smiles in zip(onehots_list, new_smiles_list)]
            else:
                tmp = [write_dict(name, onehots, smiles) for onehots, smiles in zip(onehots_list, new_smiles_list)]

        except Exception as inst:
            print(inst)
            continue
        
        new_data += tmp

    for item in new_data:
        item['onehots'] = item['onehots'].tolist()

    with open(f'data/{data_choice}/data.json', 'w') as f:
        json.dump(new_data, f)
    
    return new_data, longest_len
        

def make_1d_padding(data, longest_len):  
    onehots = np.array(data['onehots']).copy()
    new_onehots = np.zeros((onehots.shape[0], longest_len))
    left_margin = (longest_len - onehots.shape[1] + 1)//2
    right_margin = onehots.shape[1] + left_margin
    new_onehots[:, left_margin:right_margin] = onehots
    # new_onehots = new_onehots / onehots.shape[1] * 100
    
    return new_onehots



if __name__ == "__main__":
    #find_elements()
    '''
    with open('element_list.txt') as f:
        lines = f.readlines()
        elements = lines[0].split(sep=' ')[:-1]
        longest_len = int(lines[0].split(sep=' ')[-1])
        print(elements)
    '''
    '''
    pos_weight=7.0
    path = paths['valid']
    rdata = []
    names_onehots = np.load(os.path.join(path, 'names_onehots.npy'), allow_pickle=True)
    df_smiles = pd.read_csv(os.path.join(path, 'names_smiles.txt'))
    df_labels = pd.read_csv(os.path.join(path, 'names_labels.txt'))

    names_onehots = dict(names_onehots[()])
    names = names_onehots['names']
    onehots = names_onehots['onehots']
    names2 = df_smiles.loc[:, 'Chemical']
    names3 = df_labels.loc[:, 'Chemical']

    if not (type(names[0]) == str): # due to some strange error
        names = [key.decode("utf-8") for key in names]
    
    for i, key in enumerate(names):
        weight = 1.0
        if df_labels.iloc[i, 1] > 0.5:
            weight = pos_weight
        tmp = {'name': key, 'onehots': onehots[i], 'SMILES': df_smiles.iloc[i, 1], \
            'label': float(df_labels.iloc[i, 1]), 'weight': weight}
        rdata.append(tmp)

    augment_smiles(rdata)
    '''
    
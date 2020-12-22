import numpy as np
import os
import copy
import pandas as pd
import torch

paths = {'test': "../code/data/test",
        'train': "../code/data/train",
        'valid': "../code/data/validation"}


def read_data(choice='train', pos_weight=7.0, add_valid=False):
    if choice == 'train':
        return read_train_data(pos_weight, add_valid)
    elif choice == 'validation':
        return read_valid_data(pos_weight, add_valid)
    else:
        return read_test_data(pos_weight)


def read_train_data(pos_weight=7.0, add_valid=False, to_valid=False):
    path = paths['train']
    rdata = []
    names_onehots = np.load(os.path.join(path, 'names_onehots.npy'), allow_pickle=True)
    df_smiles = pd.read_csv(os.path.join(path, 'names_smiles.txt'))
    df_labels = pd.read_csv(os.path.join(path, 'names_labels.txt'))

    names_onehots = dict(names_onehots[()])
    names = names_onehots['names']
    onehots = names_onehots['onehots']

    if not (type(names[0]) == str): # due to some strange error
        names = [key.decode("utf-8") for key in names]
    
    for i, key in enumerate(names):
        weight = 1.0
        assert df_labels.iloc[i, 1] == 0 or df_labels.iloc[i, 1] == 1
        if df_labels.iloc[i, 1] > 0.5:
            weight = pos_weight
        tmp = {'name': key, 'onehots': onehots[i], 'SMILES': df_smiles.iloc[i, 1], \
            'label': float(df_labels.iloc[i, 1]), 'weight': weight}
        rdata.append(tmp)

    if add_valid: # only use 500+ as training data
        return rdata[500:]
    elif to_valid: # add 0 ~ 500 data to test set
        return rdata[:500]

    return rdata


def read_valid_data(pos_weight=7.0, add_valid=False):
    path = paths['valid']
    rdata = []
    names_onehots = np.load(os.path.join(path, 'names_onehots.npy'), allow_pickle=True)
    df_smiles = pd.read_csv(os.path.join(path, 'names_smiles.txt'))
    df_labels = pd.read_csv(os.path.join(path, 'names_labels.txt'))

    names_onehots = dict(names_onehots[()])
    names = names_onehots['names']
    onehots = names_onehots['onehots']

    if not (type(names[0]) == str): # due to some strange error
        names = [key.decode("utf-8") for key in names]
    
    for i, key in enumerate(names):
        weight = 1.0
        if df_labels.iloc[i, 1] > 0.5:
            weight = pos_weight
        tmp = {'name': key, 'onehots': onehots[i], 'SMILES': df_smiles.iloc[i, 1], \
            'label': float(df_labels.iloc[i, 1]), 'weight': weight}
        rdata.append(tmp)
    
    if add_valid:
        add_data = read_train_data(pos_weight, to_valid=True)
        rdata = rdata + add_data
    return rdata


def read_test_data(pos_weight=7.0):
    path = paths['test']
    rdata = []
    names_onehots = np.load(os.path.join(path, 'names_onehots.npy'), allow_pickle=True)
    df_smiles = pd.read_csv(os.path.join(path, 'names_smiles.txt'))

    names_onehots = dict(names_onehots[()])
    names = names_onehots['names']
    onehots = names_onehots['onehots']

    if not (type(names[0]) == str): # due to some strange error
        names = [key.decode("utf-8") for key in names]
    
    for i, key in enumerate(names):
        tmp = {'name': key, 'onehots': onehots[i], 'SMILES': df_smiles.iloc[i, 1]}
        #print(onehots[i].shape)
        rdata.append(tmp)

    return rdata


def calc_mean(data, dim1=None):
    summary = []
    dim = [1, 1]
    dim[0] = np.array(data[0]['onehots']).shape[0]

    if dim1:
        dim[1] = dim1
    else:
        dim[1] = np.array(data[0]['onehots']).shape[1]

    for item in data:
        num = np.array(item['onehots']).sum()
        summary.append(num)
    
    summary = np.array(summary)
    mean = np.mean(summary) / (dim[0] * dim[1])
    var = np.mean(summary**2) - mean**2
    var = np.sqrt(var) / (dim[0] * dim[1])

    print('mean', mean, 'var', var)
    # raise SyntaxError

    return mean, var


def sigmoid(x):
    return 1/(1+np.exp(-x))
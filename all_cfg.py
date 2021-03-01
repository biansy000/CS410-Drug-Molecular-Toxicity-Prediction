trivial_conv_cfg = {
    'model': 'TrivialConv', 
    'output_choice': 'onehots',
    'lr': 3e-3,
    'batch_size': 10, 
    'num_epoches': 35, 
    'use_LSTM': False,
    'loss_type': 'bce', 
    'data_augment': True,
    'add_valid': True, 
    'pos_weight': 8,
    'milestones': [20]
}

trivial_res_cfg = {
    'model': 'TrivialRes', 
    'num_layers': 34,
    'output_choice': 'onehots',
    'lr': 1e-3, 
    'batch_size': 500, 
    'num_epoches': 61, 
    'use_LSTM': False,
    'loss_type': 'bce', 
    'data_augment': True,
    'add_valid': False, 
    'pos_weight': 6,
    'milestones': [50],
    'exp_num': 'no_edgeweight'
}

all_cfg = {
    'trivial_conv': trivial_conv_cfg,
    'trivial_res': trivial_res_cfg
}

'''
a = [6, 7, 8, 35, 17, 11, 16, 15, 20, 9, 5, 33, 13, 53, 14, 19, 24, 30, 34, 40, 26, 50, 60, 
    29, 79, 82, 81, 51, 48, 46, 22, 1, 78, 49, 56, 47, 66, 80, 3, 70, 25, 12, 27, 28, 4, 32, 83, 23, 38, 42]
b = [6, 7, 8, 17, 35, 16, 9, 19, 44, 11, 15, 53, 3, 30]
c = [8, 6, 7, 11, 35, 15, 16, 17, 53, 3, 9, 14, 20, 19, 5, 33, 13, 24, 26, 51, 50, 80, 66, 63, 21]

total = a + b + c
my_dict = {}
cnt = 0
for item in total:
    if not item in my_dict:
        my_dict[item] = cnt
        cnt = cnt + 1
    
print(my_dict)
'''
'''
number = {6: 106498, 7: 10163, 8: 22831, 35: 458, 17: 2995, 11: 405, 16: 1929, 15: 293, 20: 26, 
    9: 1570, 5: 28, 33: 16, 13: 39, 53: 201, 14: 108, 19: 43, 24: 9, 30: 8, 34: 4, 40: 2, 26: 9, 
    50: 11, 60: 1, 29: 4, 79: 3, 82: 1, 81: 1, 51: 2, 48: 3, 46: 1, 22: 3, 1: 7, 78: 6, 49: 4, 56: 5, 
    47: 2, 66: 1, 80: 6, 3: 5, 70: 1, 25: 3, 12: 2, 27: 5, 28: 4, 4: 1, 32: 1, 83: 1, 23: 1, 38: 1, 42: 1}

keys = []
for key, val in number.items():
    if val >= 5:
        keys.append(key)

# print(keys)
idx = {}
for i, key in enumerate(keys):
    idx[key] = i

print(idx)
'''
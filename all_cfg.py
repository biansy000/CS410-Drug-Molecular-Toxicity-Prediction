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
    'batch_size': 100, 
    'num_epoches': 30, 
    'use_LSTM': False,
    'loss_type': 'bce', 
    'data_augment': True,
    'add_valid': False, 
    'pos_weight': 6.5,
    'milestones': [20],
    'exp_num': 0
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
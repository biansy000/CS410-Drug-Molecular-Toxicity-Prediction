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
    'batch_size': 300, 
    'num_epoches': 25, 
    'use_LSTM': False,
    'loss_type': 'bce', 
    'data_augment': True,
    'add_valid': False, 
    'pos_weight': 6,
    'milestones': [15]
}

all_cfg = {
    'trivial_conv': trivial_conv_cfg,
    'trivial_res': trivial_res_cfg
}
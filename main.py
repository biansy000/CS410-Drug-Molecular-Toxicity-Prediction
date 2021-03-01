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

# torch.manual_seed(1)
# dtype = torch.float32

train_dataset = Smiles(data_choice='train', pos_weight=my_cfg['pos_weight'], device=device)
valid_dataset = Smiles(data_choice='valid', pos_weight=my_cfg['pos_weight'], device=device)
train_loader = DataLoader(train_dataset, batch_size=my_cfg['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=my_cfg['batch_size'], shuffle=False)

max_waccu = 0.0

def train(model, optimizer, lr_scheduler):
    for epoch in range(my_cfg['num_epoches']):
        train_metric = EvaMetric(pos_weight=my_cfg['pos_weight'])
        train_AUC_metric = AUCMetric()
        loss_sum = 0
        model.train()

        print(f'Epoch {epoch} begins ...')

        for i, (adj, features, labels, weight) in tqdm(enumerate(train_loader)):
            res = model(features, adj)
            res = res.reshape(-1)
            # print(labels.shape)
            optimizer.zero_grad()
            if epoch > -1:
                loss = weighted_focal_loss(res, labels.squeeze(-1), weight.squeeze(-1))
            else:
                loss = weighted_bce_loss(res, labels.squeeze(-1), weight.squeeze(-1))
                
            loss.backward()
            optimizer.step()

            train_metric.update(pred=res.squeeze(-1).detach(), gt=labels)
            train_AUC_metric.update(pred=res.squeeze(-1).detach(), gt=labels)
            loss_sum += loss.detach()
        
        lr_scheduler.step()
        acc = train_metric.accuracy()
        recall = train_metric.recall()
        auc = train_AUC_metric.AUC()
        batch_size = my_cfg['batch_size']
        print(f'Acc: {acc}, Recall: {recall}, Loss: {loss_sum}, AUC: {auc}')
        
        valid(model, epoch)

    return


def valid(model, epoch):
    global max_waccu

    acc = 0.0
    valid_metric = EvaMetric(pos_weight=my_cfg['pos_weight'])
    valid_AUC_metric = AUCMetric()
    model.eval()
    print('Starting test ...')

    for i, (adj, features, labels, weight) in enumerate(valid_loader):
        res = model(features, adj)
        res = res.reshape(-1).detach()

        valid_metric.update(pred=res, gt=labels)
        valid_AUC_metric.update(pred=res.squeeze(-1), gt=labels)
            
    acc = valid_metric.accuracy()
    recall = valid_metric.recall()
    weight_accu = valid_metric.weighted_accuracy()
    auc = valid_AUC_metric.AUC()
    print(f'Acc: {acc}, Recall: {recall}, AUC: {auc}, Weighted Acc: {weight_accu}\n')

    if auc > max_waccu and epoch>10:
        max_waccu = auc
        torch.save(model.state_dict(), './bestmodel.pth')
    
    if epoch % 10 == 0 and epoch >20:
         torch.save(model.state_dict(), f'./model_{epoch}.pth')

    return 


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


def main():
    model = GCN(34, 32, 2)
    # model.load_state_dict(torch.load('model_40.pth'))
    if torch.cuda.is_available():
        model = model.cuda()
    
    # model._initialize()
    print(my_cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=my_cfg['lr'])
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=my_cfg['lr'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=my_cfg['milestones'], gamma=0.1)

    train(model, optimizer, lr_scheduler)

    return 


if __name__ == "__main__":
    #train = True
    main()
    # write_test('./bestmodel.pth')
    write_test('./model_30.pth')
    write_test('./model_40.pth')
    write_test('./model_50.pth')
    write_test('./model_60.pth')
    write_test('./model_70.pth')
    # write_test('./model_80.pth')
    # write_test('./model_90.pth')
    # write_test('./model_100.pth')
    # write_test('./model_110.pth')
    print(max_waccu)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn

class TrivialLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=8, dropout=0.05, 
        batch_first=True, bidirectional=True):

        super(TrivialLSTM, self).__init__()
        self.LSTM_model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, 
            num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

        num_direction = 2 if bidirectional==True else 1
        self.fc_layer = nn.Linear(num_direction*hidden_size, 1)
    
    def forward(self, input):
        output, (h_n, c_n) = self.LSTM_model(input)
        
        output = torch.mean(output, dim=1)
        res = self.fc_layer(output)

        likelihood = torch.sigmoid(res)

        return likelihood
        

class TrivialConv(nn.Module):
    def __init__(self, inp_height, inp_len):
        super(TrivialConv, self).__init__()
        
        self.convs = nn.Sequential(
            nn.Conv1d(inp_height, 32, padding=2, kernel_size=3), #seq len convert to inp_len + 2
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, kernel_size=4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 8, kernel_size=4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            #nn.AdaptiveMaxPool1d(inp_len//8),
            nn.Dropout(p=0.5),
            nn.Conv1d(8, 1, kernel_size=4, padding=1, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )
    
    def forward(self, input):
        out = self.convs(input)

        return out.sigmoid()


class LSTMConv(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.1, 
        batch_first=True, bidirectional=True):

        super(LSTMConv, self).__init__()
        self.LSTM_model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, 
            num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

        num_direction = 2 if bidirectional==True else 1

        self.convs = nn.Sequential(
            nn.Conv1d(num_direction*hidden_size, 64, kernel_size=4, padding=2), #seq len convert to inp_len + 2
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=4, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 8, kernel_size=4, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv1d(8, 1, kernel_size=4, padding=2, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )
    
    def forward(self, input):
        output, (h_n, c_n) = self.LSTM_model(input)
        #print('size', output.shape)
        
        output = torch.transpose(output, 1, 2)
        #assert output.shape[1] == num_direction*hidden_size

        res = self.convs(output)

        likelihood = torch.sigmoid(res)

        return likelihood   


class TrivialRes(nn.Module):
    def __init__(self, num_layers=34):
        super(TrivialRes, self).__init__()

        import torchvision.models as tm
        if num_layers == 50:
            x = tm.resnet50(pretrained=False)
            self.feature_channel = 2048
        elif num_layers == 34:
            x = tm.resnet34(pretrained=False)
            self.feature_channel = 512
        elif num_layers == 18:
            x = tm.resnet18(pretrained=False)
            self.feature_channel = 512
        else:
            raise NotImplementedError

        x.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        x.fc = nn.Linear(self.feature_channel, 1, bias=True)
        self.model = x

    def forward(self, input):
        input = input.reshape(input.shape[0], 1, input.shape[1], input.shape[2])
        out = self.model(input)
        #print(out.shape)

        return out.sigmoid()


class DeepConv(nn.Module):
    def __init__(self, inp_height, inp_len):
        super(DeepConv, self).__init__()
        
        self.convs = nn.Sequential(
            nn.Conv1d(inp_height, 64, padding=2, kernel_size=3), #seq len convert to inp_len + 2
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, kernel_size=4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            #nn.AdaptiveMaxPool1d(56),
            nn.Conv1d(16, 8, kernel_size=4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 1, kernel_size=4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.linear = nn.Linear((inp_len+2)//16, 1)
    
    def my_convs(self, inp_len, inp_channel, out_channel, out_len, kernel_size, stride):
        pad = (out_len + kernel_size - inp_len)//2
        return nn.Conv1d(inp_channel, out_channel, kernel_zize=kernel_size, padding=pad)
    
    def forward(self, input):
        out = self.convs(input)
        out = out.reshape(out.shape[0], -1)
        out = self.linear(out)

        return out.sigmoid()


class Res1d(nn.Module):
    def __init__(self, inplanes, planes, stride, outlen):
        super(Res1d, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(inplanes, planes, stride=2, padding=1, kernel_size=3) # down sample by 2
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = nn.AdaptiveAvgPool1d(outlen)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



if __name__ == "__main__":
    model = TrivialRes(18)
    print(model.model)

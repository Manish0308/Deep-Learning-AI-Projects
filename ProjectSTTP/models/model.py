    


import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['backend'] = 'SVG'
import time
import os
import datetime
import random

Nodes_num = 30

OUT_SIZE = 1 
device = "cpu"
# Matrix standardization
def normalize(A, symmetric=True):
    d = A.sum(1)
    if symmetric:
        # D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5).to(device))  # Degree matrix
        return D.mm(A).mm(D)
    else:
        # D=D^-1
        D = torch.diag(torch.pow(d, -1).to(device))
        return D.mm(A)








class SeqDataset(Dataset):
    def __init__(self, dataf, AC_martix, AD_martix, AW_data, inputnum):
        self.imgseqs = dataf  # [nodes x time]
        self.num_samples = self.imgseqs.shape[1]
        self.inputnum = inputnum
        self.inputnumT = 3 * self.inputnum
        self.AC_M = AC_martix
        self.AD_M = AD_martix
        self.AW_data = AW_data  # [nodes x time]

    def __getitem__(self, index):
        current_index = np.random.choice(range(self.inputnumT, self.num_samples))

        # Short-term temporal input
        current_imgs = self.imgseqs[:, current_index - self.inputnum:current_index]
        current_imgs = torch.FloatTensor(current_imgs)

        # Long-term temporal input (sampled every 3 steps)
        current_imgs1 = []
        for i in range(current_index - self.inputnumT, current_index, 3):
            if 0 <= i < self.num_samples:
                current_imgs1.append(self.imgseqs[:, i])
            else:
                current_imgs1.append(np.zeros((self.imgseqs.shape[0],)))
        current_imgs1 = torch.FloatTensor(current_imgs1).T

        # Label at time t
        current_label = torch.FloatTensor(self.imgseqs[:, current_index])

        # Ease-of-movement (external condition) at time t
        # ease_vector = self.AW_data[:, current_index]  # shape: [Nodes_num]
        # AW_M = np.outer(ease_vector, np.ones_like(ease_vector))  # broadcast to matrix
        # np.fill_diagonal(AW_M, 0)  # no self-loop
        # AW_M = normalize(torch.FloatTensor(AW_M).to(device), True)

        AW_data = np.random.rand(Nodes_num, Nodes_num)
        AW_M = normalize(torch.FloatTensor(AW_data).to(device), True)

        return current_imgs, current_imgs1, current_label, self.AC_M, self.AD_M, AW_M

    def __len__(self):
        return self.imgseqs.shape[1]




class SpatialFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
       # input_dim is 8 , hidden_is 64 & output is 10

        super(SpatialFeatureExtractor, self).__init__()
        self.gcn1 = nn.Linear(input_dim, hidden_dim)  # First GCN layer input_dim =8
        self.gcn2 = nn.Linear(hidden_dim, output_dim)  # Second GCN layer

        self.fc_reduce = nn.Linear(output_dim, output_dim)  # Reduce spatial dimension to match temporal output

    def forward(self, adjacency_matrix, node_features):
        """
        adjacency_matrix: The adjacency matrix (AF) of shape (batch_size, Nodes_num, Nodes_num).
        node_features: The input node features of shape (batch_size, Nodes_num, input_dim).
        """
        # adjacency matrix # AF shape : torch.Size([10, 358, 358])
        # First GCN layer

        batch_size, Nodes_num, _ = node_features.shape
        x = F.relu(self.gcn1(torch.matmul(adjacency_matrix, node_features)))
        # Second GCN layer
        x = self.gcn2(torch.matmul(adjacency_matrix, x))
        # x_len = len(x)
        # x = torch.tensor(x)
        # x = nn.Conv2d(in_channels=x_len, out_channels=Nodes_num, kernel_size=3, stride=1, padding=1)
        # print(f"inside SpatialfeatureExtractor (shape of x after gcn1&2): {x.shape}")
        # fc_reduce poye  mundu x size -> 10,358,10
        # Reduce spatial dimension to match temporal output
        x = self.fc_reduce(x)  # Shape: (batch_size, Nodes_Num, output_dim)

        return x.mean(dim=2)  # Reduce to (batch_size, output_dim)







class   TemporalModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1,output_dim=30):
        super(TemporalModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_reduce = nn.Linear(hidden_dim, self.output_dim)  # Reduce temporal output to match spatial output

    def forward(self, temporal_features):
        """
        temporal_features: Input tensor of shape (batch_size, sequence_length, input_dim)
        """
        batch_size, seq_len, _ = temporal_features.shape

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(temporal_features.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(temporal_features.device)

        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(temporal_features, (h0, c0))

        # Reduce temporal output to match spatial output
        lstm_out = self.fc_reduce(lstm_out[:, -1, :])  # Shape: (batch_size, 358)
        return lstm_out



# #Construction of ST-CGCN model


# class SiLU(nn.Module):
#     @staticmethod
#     def forward(x):
#         return x * torch.sigmoid(x)

# def autopad(k, p=None):
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p

# class Conv(nn.Module):
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
#         super(Conv, self).__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
#         self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

#     def fuseforward(self, x):
#         return self.act(self.conv(x))

import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None):  # Automatically calculate padding
    return p if p is not None else k // 2

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)

        # Use ReLU instead of SiLU
        self.act = nn.ReLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))



class STCGCN_Net(nn.Module):
    def __init__(self, dim_in, dim_out, nodes_num):
        super(STCGCN_Net, self).__init__()
        self.outlen = dim_out
        self.inlen = dim_in
        self.Nodes_num = nodes_num  # <= NEW LINE

        self.fc1 = nn.Linear(nodes_num, nodes_num)
        self.fc2 = nn.Linear(nodes_num, nodes_num)
        self.fc3 = nn.Linear(nodes_num, nodes_num)

        self.Fus1 = nn.Sequential(Conv(3, 1, 1, 1))

        self.fc4 = nn.Linear(8, 4)
        self.fc5 = nn.Linear(4, 1)

        self.spatial = SpatialFeatureExtractor(input_dim=dim_in, hidden_dim=64,output_dim=self.Nodes_num)
        self.temporal = TemporalModule(input_dim=dim_in, hidden_dim=64, output_dim=self.Nodes_num)

        self.fc6 = nn.Linear(nodes_num, nodes_num)
        self.fc7 = nn.Linear(nodes_num, nodes_num)


    def forward(self,X,X1,AC,AD,AW):
        Batch=X.shape[0]
        # print(f"X shape : {X.shape}")
        # print(f"X1 shape : {X1.shape}")
        #train loader anni batches unnavi inla pettesthadhi ante 358,8 untayi kada ala oka 10 vi theeskoni
        # okate variable la pedthadhi so anduke veeti shapes
        # 10,358
        X0=X
        #spatial
        # print("in forward function")
        AC = self.fc1(AC).view(Batch,-1,Nodes_num,Nodes_num)
        # print(AC)
        # print(f"AC shape : {AC.shape}")
        AD = self.fc2(AD).view(Batch,-1,Nodes_num,Nodes_num)
        # print(f"AD shape : {AD.shape}")
        AW = self.fc3(AW).view(Batch,-1,Nodes_num,Nodes_num)
        # print(f"AW shape : {AW.shape}")
        AF=self.Fus1(torch.cat((AC,AD,AW),1)).view(-1,Nodes_num,Nodes_num) #矩阵融合
        # print(f"AF shape : {AF.shape}. X:shape :{X.shape}") # AF shape : torch.Size([10, 358, 358])
        # print(AF)
        #spatial block

        XS=self.spatial(AF,X) #Replace with your spatial feature extraction module  X shape : torch.Size([10, 358, 8]) short seq
        # print(f"XS shape : {XS.shape}")
        # print(XS)
        #temporal block
        XT=self.temporal(X1) #Replace with your temporal feature extraction module X1 shape : torch.Size([10, 358, 8]) long seq.
        # print(f"XT shape : {XT.shape}")
        # print(XT)
        #Fuse
        # print(XS.shape)
        # print(XT.shape)
        out=torch.relu(self.fc6(XS)+self.fc7(XT))
        print(f"out shape is {out.shape}")
        print(f"out is {out}")

        out = out.view(Batch, Nodes_num, OUT_SIZE)
        print(f"out shape is {out.shape}")
        return out

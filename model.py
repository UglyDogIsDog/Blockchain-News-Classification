import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):
    
    def __init__(self):
        super(CNN_Text, self).__init__()
        Co = 100 # number of kernel
        Ks = [3, 4, 5] # size of kernels, number of features
        Dropout = 0.5

        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, 768)) for K in Ks])
        self.dropout = nn.Dropout(Dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, 2)
        #self.act = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W+-), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        res = self.fc1(x)  # (N, C)
        #res = self.act(res)
        return res

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import getopt
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import json
import numpy as np

class CNN_Text(nn.Module):
    def __init__(self):
        super(CNN_Text, self).__init__()
        Co = 100 # number of kernel
        Ks = [3, 4, 5] # size of kernels, number of features
        Dropout = 0.5

        #self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, 768)) for K in Ks])
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, 768)) for K in Ks])
        self.conv2 = nn.Conv1d(100,50,5,stride = 4)
        self.conv3 = nn.Conv1d(50,20,5)#in_channels,out_channels,kernel_size
        self.dropout = nn.Dropout(Dropout)
        self.fc1 = nn.Linear(60, 2)
        #self.act = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W+-), ...]*len(Ks)
        x = [self.dropout(i) for i in x]
        x = [F.max_pool1d(i, 5,stride = 2) for i in x]  # [(N, Co), ...]*len(Ks)
        #add another conv_layer&pool_layer
        x = [F.relu(self.conv2(i)) for i in x]
        x = [self.dropout(i) for i in x]
        x = [F.relu(self.conv3(i)) for i in x]
        x = [self.dropout(i) for i in x]
        
        x = [F.max_pool1d(i,i.shape[2]) for i in x]
      
        x = torch.cat(x, 1).squeeze(2)
        
        x = self.dropout(x)  # (N, len(Ks)*Co)
        res = self.fc1(x)  # (N, C)
        return res
        
        

#test
def test(cnn, test_loader, use_cuda):
    pred_v = []
    right, total = 0, 0
    right_neg, total_neg = 0, 0
    right_pos, total_pos = 0, 0
    for step, data in enumerate(test_loader):
        
        vec, lens, label = data
        #print(vec.shape)
        if use_cuda:
            vec = vec.cuda()
            label = label.cuda()
        output = cnn(vec)
        pred = torch.max(output,1)[1]
        pred_v.extend(pred)
        label = label.to(dtype=torch.int64)
        
        right_neg += label[(pred == label) & (label == 0)].size(0)
        total_neg += label[label == 0].size(0)
        right_pos += label[(pred == label) & (label == 1)].size(0)
        total_pos += label[label == 1].size(0)
        right += label[pred == label].size(0)
        total += label.size(0)
    print('Accuracy:%.3f %d/%d' % (float(right_neg + right_pos) / float(total_neg + total_pos), right_neg + right_pos, total_neg + total_pos))
    print('Negative accuracy:%.3f  %d/%d' % (float(right_neg) / float(total_neg), right_neg, total_neg))
    print('Positive accuracy:%.3f  %d/%d' % (float(right_pos) / float(total_pos), right_pos, total_pos))
    #print(len(pred_v))
    #print("predict positive number:",pred_v[pred_v==1].size(0),"predict negative number:",pred_v[pred_v==0].size(0))
    #torch.save(pred_v,"test.json.lab")
    #outp = open("pred_label.json", 'w', encoding="utf-8")
    #outp.write(json.dumps(list(np.array(pred_v)), indent=4, ensure_ascii=False))
    #outp.close()

#model
'''class CNN(nn.Module):
    def __init__(self, input_size , out_class):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, 5, padding=2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 32, 5, padding=2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(32, 16, 5, padding=2),
            nn.ReLU()
        )
        #pool_layer
        self.pool = nn.Sequential(
           nn.MaxPool1d(4,stride = 2,padding = 2)
        )
        # fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 9, 512),
        	nn.ReLU(),
        	nn.Dropout(),
        	nn.Linear(512, out_class),
            #nn.Sigmoid()
        )
    def forward(self, x):
        out = self.conv(x)
        #池化层
        out = out.permute(0,2,1)
        out = self.pool(out)
        out = out.permute(0,2,1)
        out = out.contiguous()
        
        out = out.view(out.size(0), -1) #unfold
        out = self.fc(out)
        return out
'''
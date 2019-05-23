import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#output = torch.randn(100,128,768)#测试用例
import sys
import getopt
import torch
import torch.utils.data as Data
import torch.nn.functional as F
#output = torch.randn(100,128,768)#测试用例
class CNN_Text(nn.Module):
    def __init__(self):
        super(CNN_Text, self).__init__()
        Co = 100 # number of kernel
        Ks = [3, 4, 5] # size of kernels, number of features
        Dropout = 0.5
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, 768//4),stride = (K,768//4)) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d((64//K)//2,1,5,stride = 5) for K in Ks])
        self.dropout = nn.Dropout(Dropout)
        self.fc1 = nn.Linear(len(Ks)*Co//5, 2)
        

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)) for conv in self.convs1]  # [(N, Co, W+-), ...]*len(Ks)
        x = [self.dropout(i) for i in x]
        
        x = [F.max_pool2d(i, (2,i.size(3))).squeeze(3) for i in x]  # [(N, Co), ...]*len(Ks)
        #add another conv_layer
        #change_dim
        x = [i.permute(0,2,1) for i in x]
        x = [F.relu(self.convs2[i](x[i])) for i in range(3)]
        x = [self.dropout(i) for i in x]
        x = [i.permute(0,2,1) for i in x]
        
        x = torch.cat(x, 1).squeeze(2)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        res = self.fc1(x)  # (N, C)
        return res
        
        #return x
        

#test
def test(cnn, test_loader, use_cuda):
    right, total = 0, 0
    right_neg, total_neg = 0, 0
    right_pos, total_pos = 0, 0
    for step, data in enumerate(test_loader):
        vec, label = data
        if use_cuda:
            vec = vec.cuda()
            label = label.cuda()
        output = cnn(vec)
        label = label.to(dtype=torch.int64)
        
        pred = torch.max(output, 1)[1]
        right_neg += label[(pred == label) & (label == 0)].size(0)
        total_neg += label[label == 0].size(0)
        right_pos += label[(pred == label) & (label == 1)].size(0)
        total_pos += label[label == 1].size(0)
        right += label[pred == label].size(0)
        total += label.size(0)
    print('Accuracy:%.3f %d/%d' % (float(right_neg + right_pos) / float(total_neg + total_pos), right_neg + right_pos, total_neg + total_pos))
    print('Negative accuracy:%.3f  %d/%d' % (float(right_neg) / float(total_neg), right_neg, total_neg))
    print('Positive accuracy:%.3f  %d/%d' % (float(right_pos) / float(total_pos), right_pos, total_pos))


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
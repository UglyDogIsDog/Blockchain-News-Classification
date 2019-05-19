import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
import torchvision
import json
import bert_encoder
import os
import random

# Hyper Parameters
print("input epoch")
EPOCH = input()
BATCH_SIZE = 50
LR = 1e-3

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True

#customized loading data
class CustomDataset(Dataset):
    def __init__(self, path):
        if os.path.isfile(path + ".dat") and os.path.isfile(path + ".lab"):    
            self.data = torch.load(path + ".dat")
            self.label = torch.load(path + ".lab")
            return

        inp = open(path, "rb")
        passages = json.load(inp)
        self.data = []
        self.label = []
        be = bert_encoder.BertEncoder()
        pos_num, neg_num, num = 0, 0, 0
        pos_index = []
        neg_index = []
        for passage in passages:
            print(len(passage["passage"]))
            while len(passage["passage"]) > 32: #abandon too short section
                self.data.append(torch.FloatTensor(be.encode(passage["passage"][:128])).squeeze(0).transpose(0, 1))
                if passage["label"] == 1:
                    self.label.append(torch.FloatTensor([1]))
                    pos_num += 1
                    pos_index.append(num)
                else:
                    self.label.append(torch.FloatTensor([0]))
                    neg_num += 1
                    neg_index.append(num)
                num += 1
                passage["passage"] = passage["passage"][128:]
        inp.close()

        while pos_num < neg_num:
            self.data.append(self.data[pos_index[random.randint(0, len(pos_index) - 1)]].clone())
            self.label.append(torch.FloatTensor([1]))
            pos_num += 1

        while pos_num > neg_num:
            self.data.append(self.data[neg_index[random.randint(0, len(neg_index) - 1)]].clone())
            self.label.append(torch.FloatTensor([0]))
            neg_num += 1
        
        
        torch.save(self.data, path + ".dat")
        torch.save(self.label, path + ".lab")

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

train_loader = Data.DataLoader(dataset = CustomDataset("train.json"), batch_size = BATCH_SIZE, shuffle = True)
test_loader = Data.DataLoader(dataset = CustomDataset("test.json"), batch_size = BATCH_SIZE, shuffle = True)

#model
class CNN(nn.Module):
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
           nn.MaxPool1d(2)
        )
        # fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 8, 128),
        	nn.ReLU(),
        	nn.Dropout(),
        	nn.Linear(128, out_class),
            nn.Sigmoid()
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

cnn = CNN(768, 1) #bert output a vector of 768 for every word, 
                  #and the output mental analysis is binary classification
if use_cuda:
    cnn = cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(),lr = LR)
loss_func = nn.BCELoss() #for float

#train
for epoch in range(EPOCH):
    for step, data in enumerate(train_loader):
        vec, label = data
        if use_cuda:
            vec = vec.cuda()
            label = label.cuda()
        output = cnn(vec)
        #print(output)
        loss = loss_func(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #output process every 100 batch
        if step % 1 == 0:
            output = output - label #count right answer
            accuracy = float(output[((output >= -0.5) & (output <= 0.5))].size(0)) / float(label.size(0))
            print('Epoch:', epoch, '|| Loss:%.4f' % loss, '|| Accuracy:%.3f' % accuracy)

#test
#right, total = 0, 0
right_neg, total_neg = 0, 0
right_pos, total_pos = 0, 0
for step, data in enumerate(test_loader):
    vec, label = data
    if use_cuda:
        vec = vec.cuda()
        label = label.cuda()
    #print(vec)
    output = cnn(vec)
    #print(output)
    #output = output - label
    #print(output)
    right_neg += output[(label < 0.5) & (output < 0.5)].size(0)
    total_neg += label[(label < 0.5)].size(0)
    right_pos += output[(label > 0.5) & (output > 0.5)].size(0)
    total_pos += label[(label > 0.5)].size(0)
    #right += output[((output >= -0.5) & (output <= 0.5))].size(0)
    #total += label.size(0)

print('Accuracy:%.3f' % (float(right_neg + right_pos) / float(total_neg + total_pos)))
#print(right, " ", total)
print('Negative accuracy:%.3f' % (float(right_neg) / float(total_neg)))
#print(right_neg, " ", total_neg)
print('Positive accuracy:%.3f' % (float(right_pos) / float(total_pos)))
#print(right_pos, " ", total_pos)
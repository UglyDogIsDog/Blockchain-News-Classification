import json
import os
import random
import sys
import numpy as np
import re
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from bert_serving.client import BertClient
import model

# Hyper Parameters
if len(sys.argv) <= 2:
    print("Error: input batch size, LR first")
    sys.exit()

EPOCH = 100
BATCH_SIZE = int(sys.argv[1])
LR = float(sys.argv[2])

CLIENT_BATCH_SIZE = 4096
SEN_NUM = 32
SEN_LEN = 32

#use CUDA to speed up
use_cuda = torch.cuda.is_available()

#cut paragraph to sentences
def cut_para(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.strip()  # remove both sizes' blanks
    return para.split("\n")

#customized loading data
class CustomDataset(Dataset):
    def __init__(self, path):
        if os.path.isfile(path + ".dat") and os.path.isfile(path + ".lab"):    
            self.data = torch.load(path + ".dat")
            self.label = torch.load(path + ".lab")
            return

        be = BertClient(ip='192.168.120.125')
        
        #read data
        inp = open(path, "rb")
        passages = json.load(inp)
        sens = []
        self.label = []
        #pos_num, neg_num = 0, 0
        #pos_index = []
        #neg_index = []
        for passage in passages:
            pass_sen = cut_para(passage["passage"])
            print(len(pass_sen))
            if len(pass_sen) < SEN_NUM:
                pass_sen += ["x"] * (SEN_LEN - len(pass_sen))
            sens += pass_sen[0: SEN_NUM]
            
            if passage["label"] == 1:
                self.label += [1]
                #pos_num += len(pass_sen)
                #pos_index += [i for i in range(len(sens), len(sens) + len(pass_sen))]
            else:
                self.label += [0]
                #neg_num += len(pass_sen)
                #neg_index += [i for i in range(len(sens), len(sens) + len(pass_sen))]
        inp.close()
            
        #send sentences to BERT-as-service, get each sentences' vector of size 768
        self.data = np.empty((len(sens), 768), dtype=np.float32)
        last_num = 0
        while len(sens) > last_num:
            start = last_num
            end = min(last_num + CLIENT_BATCH_SIZE, len(sens))
            self.data[start : end] = be.encode(sens[start : end])
            last_num = end
        
        #reshape the data for every passage
        self.data = np.resize(self.data, ((len(self.data) // SEN_NUM), SEN_NUM, 768))

        '''
        #balance the data
        while pos_num < neg_num:
            self.data[last_num] = np.copy(self.data[pos_index[random.randint(0, len(pos_index) - 1)]])
            #self.data = np.append(self.data, np.expand_dims(np.copy(self.data[pos_index[random.randint(0, len(pos_index) - 1)]]), 0), axis = 0)
            self.label.append(1)
            pos_num += 1
            last_num += 1

        while pos_num > neg_num:
            self.data[last_num] = np.copy(self.data[neg_index[random.randint(0, len(neg_index) - 1)]])
            #self.data = np.append(self.data, np.expand_dims(np.copy(self.data[pos_index[random.randint(0, len(pos_index) - 1)]]), 0), axis = 0)
            #self.data.append(self.data[neg_index[random.randint(0, len(neg_index) - 1)]].clone())
            self.label.append(0)
            neg_num += 1
            last_num += 1
        '''
        torch.save(torch.FloatTensor(self.data), path + ".dat")
        torch.save(self.label, path + ".lab")

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]

train_loader = Data.DataLoader(dataset = CustomDataset("train.json"), batch_size = BATCH_SIZE, shuffle = True)
test_loader = Data.DataLoader(dataset = CustomDataset("test.json"), batch_size = BATCH_SIZE, shuffle = True)

#initialize model
cnn = model.CNN_Text()
if use_cuda:
    cnn = cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(),lr = LR)

#test
def test():
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

#train
for epoch in range(EPOCH):
    if epoch % 5 == 0:
        test()
    for step, data in enumerate(train_loader):
        vec, label = data
        if use_cuda:
            vec = vec.cuda()
            label = label.cuda()
        output = cnn(vec)
        label = label.to(dtype=torch.int64)
        loss = F.cross_entropy(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #output process every 100 batch
        if step % 1000 == 0:
            pred = torch.max(output, 1)[1]
            accuracy = float(label[pred == label].size(0)) / float(label.size(0))
            print('Epoch:', epoch, '|| Loss:%.4f' % loss, '|| Accuracy:%.3f' % accuracy)

test()
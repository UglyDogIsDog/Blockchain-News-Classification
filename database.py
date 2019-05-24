import os
import json
import re
import random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from bert_serving.client import BertClient

CLIENT_BATCH_SIZE = 4096
SEN_NUM = 64

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
    def __init__(self, path, balance):
        if os.path.isfile(path + ".dat") and os.path.isfile(path + ".lab"):    
            self.data = torch.load(path + ".dat")
            self.label = torch.load(path + ".lab")
            return
        

        be = BertClient()
        
        #read data
        inp = open(path, "rb")
        passages = json.load(inp)
        sens = []
        self.label = []
        pos_num, neg_num = 0, 0
        pos_index = []
        neg_index = []
        for passage in passages:
            pass_sen = cut_para(passage["passage"])
            if len(pass_sen) < SEN_NUM:
                pass_sen += ["x"] * (SEN_NUM - len(pass_sen))
            pass_sen = pass_sen[0: SEN_NUM]
            sens += pass_sen
            
            if passage["label"] == 1:
                pos_num += 1
                pos_index += [len(self.label)]
                self.label += [1]
            else:
                neg_num += 1
                neg_index += [len(self.label)]
                self.label += [0]
        inp.close()
            
        #send sentences to BERT-as-service, get each sentences' vector of size 768
        if balance:
            self.data = np.empty((len(sens) + abs(neg_num - pos_num) * SEN_NUM, 768), dtype=np.float32)
        else:
            self.data = np.empty((len(sens), 768), dtype=np.float32)
        last_num = 0
        while len(sens) > last_num:
            start = last_num
            end = min(last_num + CLIENT_BATCH_SIZE, len(sens))
            self.data[start : end] = be.encode(sens[start : end])
            last_num = end
            print("%s got %d/%d" % (path, last_num, len(sens)))
        #reshape the data for every passage
        self.data = np.resize(self.data, ((len(self.data) // SEN_NUM), SEN_NUM, 768))
       
        #balance the data
        last_num = last_num // SEN_NUM
        if balance:
            while pos_num < neg_num:
                self.data[last_num] = np.copy(self.data[pos_index[random.randint(0, len(pos_index) - 1)]])
                self.label.append(1)
                pos_num += 1
                last_num += 1

            while pos_num > neg_num:
                self.data[last_num] = np.copy(self.data[neg_index[random.randint(0, len(neg_index) - 1)]])
                self.label.append(0)
                neg_num += 1
                last_num += 1
        
        torch.save(torch.FloatTensor(self.data), path + ".dat")
        torch.save(self.label, path + ".lab")

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]
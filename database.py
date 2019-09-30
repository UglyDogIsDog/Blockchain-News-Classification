import os
import json
import re
import random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from bert_serving.client import BertClient
import nltk
CLIENT_BATCH_SIZE = 4096
#SEN_NUM = 40 #max sequence length bert can receive
MIN_SEN_LEN = 10

#cut paragraph to sentences
#def cut_para(para):
#    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
#    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
#    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
#    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
#    para = para.strip()  # remove both sizes' blanks
#    return [sen.strip() for sen in para.split("\n") if len(sen.strip()) >= MIN_SEN_LEN]

def cut_para(para):
    nltk.download('punkt')#for tokenize
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(para)
    sens = [sen.strip() for sen in sentences if len(sen.strip())>=MIN_SEN_LEN]
    return sens

'''
begin/end use as label to divide paras@@
'''

#customized loading data
class CustomDataset(Dataset):
    def __init__(self, path, sen_num,train_model = True):
        if os.path.isfile(path + ".dat"):    
            self.data = torch.load(path + ".dat")
            self.label = torch.load(path + ".lab")
#            self.start = torch.load(path + ".sta")
#            self.end = torch.load(path + ".end")
            self.lens = torch.load(path + '.lens')
            self.cum_len = [0] + list(np.cumsum(np.array(self.lens)))
            self.sens = torch.load(path + '.sens')
            self.sen_num = sen_num
            return
        

        be = BertClient()

        self.sen_num = sen_num
        #read data
        inp = open(path, "rb")
        passages = json.load(inp)
        self.sens = []
        self.label,self.lens = [],[]
#        self.start = []
#        self.end = []
        #pos_num, neg_num = 0, 0
        #pos_index = []
        #neg_index = []
        for passage in passages:
            pass_sen = cut_para(passage["passage"])
            if len(pass_sen) == 0:
                pass_sen = ['x']
            if len(pass_sen) > sen_num:
                pass_sen = pass_sen[0 : sen_num]
#            self.start += [len(sens)]
            self.sens += pass_sen
      
#            self.end += [len(sens)]
            self.lens += [len(pass_sen)]
            print(len(pass_sen))
            if train_model:
                self.label += [passage["label"]]
            else:
                self.label += [0]
        inp.close()
        print('begin incoding')
        self.data = be.encode(self.sens) #every senetcne an encoding vector
        self.data = torch.FloatTensor(self.data)
        
        torch.save(self.data, path + ".dat")
        torch.save(self.label, path + ".lab")
        torch.save(self.lens,path + '.lens')
        torch.save(self.sens,path + '.sens')
        self.cum_len = [0] + list(np.cumsum(np.array(self.lens)))
#        torch.save(self.start, path + ".sta")
#        torch.save(self.end, path + ".end")

        '''
            if len(pass_sen) < SEN_NUM:
                pass_sen += ["x"] * (SEN_NUM - len(pass_sen))
            for i in range(SEN_NUM):
                if not pass_sen[i]:
                    pass_sen[i] = "x"
            pass_sen = pass_sen[0: SEN_NUM]
            sens += pass_sen
            
            if 'label' in passage.keys():
                if passage["label"] == 1:
                    pos_num += 1
                    pos_index += [len(self.label)]
                    self.label += [1]
                else:
                    neg_num += 1
                    neg_index += [len(self.label)]
                    self.label += [0]
            else:
                neg_num += 1
                neg_index += [len(self.label)]
                self.label += [0]
        '''

        
        '''
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
        '''
        

    def __getitem__(self, index):
#        cum_len = list(np.cumsum(np.array(self.lens)))
#        cum_len = [0] + cum_len #add zero as the 1st ele
        _len =  self.lens[index]
        para = self.data[self.cum_len[index]:(self.cum_len[index]+ _len)]
        _sen_num = self.sen_num
        if _len < _sen_num: #add padding
            para = torch.cat((para,torch.zeros((_sen_num - _len),768)),dim = 0)
    
#        if self.end[index] - self.start[index] <= SEN_NUM:
#            para = self.data[self.start[index] : self.end[index]]
#            length = self.end[index] - self.start[index]
#            para = torch.cat((para, torch.zeros((SEN_NUM - (self.end[index] - self.start[index]), 768))), dim=0)
#            #print(para.shape)
#        else:
#            start = random.randint(self.start[index], self.end[index] - SEN_NUM)
#            #print(start)
#            end = start + SEN_NUM
#            length = SEN_NUM
#            para = self.data[start : end]
            
        return para, _len, self.label[index] 

    def __len__(self):
        return len(self.label)
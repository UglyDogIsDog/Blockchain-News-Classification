import sys
import argparse
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from database import CustomDataset, SEN_NUM
#from model import CNN_Text, test

#BERT_MAX_SEQ_LEN = 64
CHECK_TIME = 3

class LSTM_model(nn.Module):
    def __init__(self):
        super(LSTM_model, self).__init__()
        self.lstm = nn.LSTM(768, args.hidden_layer, num_layers=2, bidirectional=True)
        self.pooling = nn.MaxPool1d(SEN_NUM)

    def forward(self, sens, lens):
        lens, perm_idx = lens.sort(0, descending=True)
        sens = sens[perm_idx]
        sens = sens.permute(1, 0, 2) # B * L * V -> L * B * V
        sens = pack_padded_sequence(sens, lens, batch_first=False, enforce_sorted=True)
        o, (h, c) = self.lstm(sens) # o: L * B * 2V
        o = pad_packed_sequence(o, batch_first=False, padding_value=0.0, total_length=None)[0] # <L * B * 2V

        h = self.pooling(o.permute(1, 2, 0)).squeeze(2) # B * 2V
        _, unperm_idx = perm_idx.sort(0)
        h = h[unperm_idx]
        return h

class MLP_model(nn.Module):
    def __init__(self):
        super(MLP_model, self).__init__()
        self.linear1 = nn.Linear(args.hidden_layer * 2, args.hidden_layer * 2) 
        self.linear2 = nn.Linear(args.hidden_layer * 2, 2)
        #self.linear3 = nn.Linear(50, 2)
        self.dropout = nn.Dropout(0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, f):
        f = F.relu(self.dropout(F.relu(self.linear1(f))))
        #f = self.dropout(F.relu(self.linear2(f)))
        f = self.linear2(f)
        return f #self.softmax(f)
'''
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        #print(transposed_data)
        self.inp = torch.stack(transposed_data[0], 0)
        #print(self.inp.shape)
        self.tgt = torch.tensor(transposed_data[1])
        #print(self.tgt.shape)

    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)
'''
if __name__ == "__main__":  
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-r", "--regularization", type=float, default=0.003) #normally 0.0005

    # relatively loose hyperparameters
    parser.add_argument("-e", "--epoch", type=int, default=500)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-c", "--clip", type=float, default=1)
    parser.add_argument("-hl", "--hidden_layer", type=int, default=50)
    parser.add_argument("-de", "--decay_epoch", type=int, default=20)
    args = parser.parse_args()

    #use CUDA to speed up
    use_cuda = torch.cuda.is_available()

    #get data
    train_loader = Data.DataLoader(dataset=CustomDataset(path="train.json", balance=False), batch_size = args.batch_size, shuffle = True)#, collate_fn=collate_wrapper, pin_memory=True)
    dev_loader = Data.DataLoader(dataset=CustomDataset(path="test.json", balance=False), batch_size = args.batch_size, shuffle = False)#, collate_fn=collate_wrapper, pin_memory=True)

    #initialize model
    lstm = LSTM_model()
    mlp = MLP_model()
    if use_cuda:
        lstm = lstm.cuda()
        mlp = mlp.cuda()
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=args.regularization)

    def run(data_loader, update_model):
        total_num = 0
        right_num = 0
        true_pos = 0
        pos = 0
        true = 0
        total_loss = 0

        iteration = 1 if update_model else CHECK_TIME

        for ite in range(iteration):
            for step, data in enumerate(data_loader):
                sens, lens, labels = data
                labels = labels.to(dtype=torch.int64)
                if use_cuda:
                    sens = sens.cuda()
                    lens = lens.cuda()
                    labels = labels.cuda()
                h = lstm(sens, lens)
                score = mlp(h) # B * Labels

                if update_model:
                    loss = F.cross_entropy(score, labels)
                    total_loss += loss
                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(lstm.parameters(), args.clip)
                    optimizer.step()
                
                if step == 0:
                    pred = torch.max(score, 1)[1]
                    targ = labels
                else:
                    pred = torch.cat((pred, torch.max(score, 1)[1]), dim=0)
                    targ = torch.cat((targ, labels), dim=0)
            if ite == 0:
                pred_sum = pred
            else:
                pred_sum += pred
        
        pred = torch.zeros(pred.shape).to(dtype=torch.int64)
        if use_cuda:
            pred = pred.cuda()
        pred[pred_sum > (CHECK_TIME * 1.0 / 2)] = 1
        labels = targ

        right_num += labels[pred == labels].size(0)
        total_num += labels.size(0)
        true_pos += labels[(labels == 1) & (pred == labels)].size(0)
        pos += labels[pred == 1].size(0)
        true += labels[labels == 1].size(0)

            
        if update_model:
            print("train: loss: {} ".format(total_loss), end="")
        else:
            print("dev: ", end="")
        accuracy = float(right_num) / total_num
        
        print("accuracy: {} ".format(accuracy), end="")
        if pos > 0:
            precision = float(true_pos) / pos
            print("precision: {} ".format(precision), end="")
        if true > 0:
            recall = float(true_pos) / true
            print("recall: {} ".format(recall), end="")
        if pos > 0 and true > 0 and (precision + recall) > 0:
            F1 = 2.0 * precision * recall / (precision + recall)
            print("F1: {} ".format(F1))
        else:
            print()

    #train
    for epoch in range(args.epoch):
        print("epoch:{}".format(epoch + 1))
        run(data_loader=train_loader, update_model=True)
        run(data_loader=dev_loader, update_model=False)
        #if (epoch + 1) % args.decay_epoch == 0:
        #    learning_rate /= 2
        #    print("lr: {}".format(learning_rate))
        #    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=args.regularization)
'''
    #train
    for epoch in range(args.epoch):
        #if epoch % 5 == 0:
        #    test(cnn, test_loader, use_cuda)
        for step, data in enumerate(train_loader):
            vec, label = data
            if use_cuda:
                vec = vec.cuda()
                label = label.cuda()
            h = lstm(vec)
            score = mlp(h)
            pred = torch.max(score, 1)[1]
            print(h.shape)
            
            label = label.to(dtype=torch.int64)

            right_num += labels[pred == labels].size(0)
            total_num += labels.size(0)
            true_pos += labels[(labels == 1) & (pred == labels)].size(0)
            pos += labels[pred == 1].size(0)
            true += labels[labels == 1].size(0)

            if update_model:
                loss = F.cross_entropy(score, labels)
                #loss = F.mse_loss(manhattan_distance, labels.float()) # mse_loss or l1_loss?
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(lstm.parameters(), args.clip)
                optimizer.step() '''

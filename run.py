import sys
import argparse
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn

from database import CustomDataset
from model import CNN_Text, test

BERT_MAX_SEQ_LEN = 64

class LSTM_model(nn.Module):
    def __init__(self):
        super(LSTM_model, self).__init__()
        self.lstm = nn.LSTM(768, args.hidden_layer, num_layers=2, bidirectional=True)
        self.pooling = nn.MaxPool1d(BERT_MAX_SEQ_LEN)

    def forward(self, sens): #, lens):
        #lens, perm_idx = lens.sort(0, descending=True)
        #sens = sens[perm_idx]
        sens = sens.permute(1, 0, 2) # B * L * V -> L * B * V
        #sens = pack_padded_sequence(sens, lens, batch_first=False, enforce_sorted=True)
        o, (h, c) = self.lstm(sens) # o: L * B * 2V
        #o = pad_packed_sequence(o, batch_first=False, padding_value=0.0, total_length=None)[0] # <L * B * 2V

        h = self.pooling(o.permute(1, 2, 0)).squeeze(2) # B * 2V
        #_, unperm_idx = perm_idx.sort(0)
        #h = h[unperm_idx]
        return h

if __name__ == "__main__":  
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-r", "--regularization", type=float, default=0.0005) #normally 0.0005

    # relatively loose hyperparameters
    parser.add_argument("-e", "--epoch", type=int, default=500)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-c", "--clip", type=float, default=1)
    parser.add_argument("-hl", "--hidden_layer", type=int, default=300)
    args = parser.parse_args()

    #use CUDA to speed up
    use_cuda = torch.cuda.is_available()

    #get data
    train_loader = Data.DataLoader(dataset = CustomDataset(path="train.json", balance=True), batch_size = args.batch_size, shuffle = True)
    test_loader = Data.DataLoader(dataset = CustomDataset(path="test.json", balance=False), batch_size = args.batch_size, shuffle = False)

    #initialize model
    lstm = LSTM_model()
    if use_cuda:
        lstm = lstm.cuda()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=args.learning_rate, weight_decay=args.regularization)


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
            print(h.shape)
            
            label = label.to(dtype=torch.int64)

            '''
            loss = F.cross_entropy(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
   
            #output process every 1000 batch
            if step % 1000 == 0:
                pred = torch.max(output, 1)[1]
                accuracy = float(label[pred == label].size(0)) / float(label.size(0))
                print('Epoch:', epoch, '|| Loss:%.4f' % loss, '|| Accuracy:%.3f' % accuracy)

    test(cnn, test_loader, use_cuda)'''
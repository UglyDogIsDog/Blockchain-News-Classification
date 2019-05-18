import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
import torchvision
import json
import bert_encoder

# Hyper Parameters
EPOCH = 100
BATCH_SIZE = 50
LR = 1e-4

#customized loading data
class CustomDataset(Dataset):
    def __init__(self, path):
        inp = open(path, "rb")
        passages = json.load(inp)
        self.data = []
        self.label = []
        be = bert_encoder.BertEncoder()
        positive = torch.FloatTensor([1])
        negative = torch.FloatTensor([0])
        for passage in passages:
            while len(passage["passage"]) > 32: #abandon too short section
                self.data.append(torch.FloatTensor(be.encode(passage["passage"][:128])).squeeze(0).transpose(0, 1))
                if passage["label"] == 1:
                    self.label.append(positive)
                else:
                    self.label.append(negative)
                passage["passage"] = passage["passage"][128:]
        inp.close()

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
        # fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 16, 128),
        	nn.ReLU(),
        	nn.Dropout(),
        	nn.Linear(128, out_class),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1) #unfold
        out = self.fc(out)
        return out

cnn = CNN(768, 1) #bert output a vector of 768 for every word, 
                  #and the output mental analysis is binary classification
optimizer = torch.optim.Adam(cnn.parameters(),lr = LR)
loss_func = nn.BCELoss() #for float

#train
for epoch in range(EPOCH):
    for step, data in enumerate(train_loader):
        vec, label = data
        output = cnn(vec)
        loss = loss_func(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #output process every 100 batch
        if step % 1 == 0:
            output = output - label #count right answer
            accuracy = float(output[((output >= -0.5) & (output <= 0.5))].size(0)) / float(label.size(0))
            print('Epoch:', epoch, '|| Loss:%.4f' % loss, '|| Accuracy:%.2f' % accuracy)

#test
for step, data in enumerate(test_loader):
    vec, label = data
    output = cnn(vec)
    if step % 1 == 0:
        output = output - label
        accuracy = float(output[((output >= -0.5) & (output <= 0.5))].size(0)) / float(label.size(0))
        print('Accuracy:%.2f' % accuracy)
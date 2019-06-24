import sys
import getopt
import torch
import torch.utils.data as Data
import torch.nn.functional as F

from database import CustomDataset
from model import CNN_Text, test

if __name__ == "__main__":  
    # Hyper Parameters
    EPOCH = 200
    BATCH_SIZE = 32
    LR = 1e-3
    opts, args = getopt.getopt(sys.argv[1:],'-b:-l:-e:', ['batch_size=', 'learning_rate=', 'epoch='])
    for opt_name, opt_value in opts:
        if opt_name in ('-b', '--batch_size'):
            BATCH_SIZE = int(opt_value)
        if opt_name in ('-l', '--learning_rate'):
            LR = float(opt_value)
        if opt_name in ('-e','--epoch'):
            EPOCH = int(opt_value)

    #use CUDA to speed up
    use_cuda = torch.cuda.is_available()

    #get data
    train_loader = Data.DataLoader(dataset = CustomDataset(path="train.json", balance=False), batch_size = BATCH_SIZE, shuffle = True)
    test_loader = Data.DataLoader(dataset = CustomDataset(path="test.json", balance=False), batch_size = BATCH_SIZE, shuffle = True)

    #initialize model
    cnn = CNN_Text()
    if use_cuda:
        cnn = cnn.cuda()
    optimizer = torch.optim.Adam(cnn.parameters(), lr = LR, weight_decay=0.0005)

    #train
    for epoch in range(EPOCH):
        print("epoch :")
        if epoch % 5 == 0:
            test(cnn, test_loader, use_cuda)
        for step, data in enumerate(train_loader):
            vec, lens, label = data
            #print(vec.shape)
            if use_cuda:
                vec = vec.cuda()
                label = label.cuda()
            output = cnn(vec)
            label = label.to(dtype=torch.int64)
            loss = F.cross_entropy(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
   
            #output process every 1000 batch
            if step % 1000 == 0:
                pred = torch.max(output, 1)[1]
                accuracy = float(label[pred == label].size(0)) / float(label.size(0))
                print('Epoch:', epoch, '|| Loss:%.4f' % loss, '|| Accuracy:%.3f' % accuracy)

    test(cnn, test_loader, use_cuda)
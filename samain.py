import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import codecs
import numpy as np
import torch.nn as nn
from visdom import Visdom
torch.cuda.set_device(0)


class vocab():
    def __init__(self, vocab_path):
        self.id2word = {}
        self.word2id = {}
        with codecs.open(vocab_path, "r", "utf8") as f:
            for line in f.readlines():
                word = line.split(" ")[1].replace("\n", "")
                id = line.split(" ")[0]
                self.id2word.update({int(id):word})
                self.word2id.update({word:int(id)})
        self.id2word.update({len(self.word2id):"<unk>"})
        self.id2word.update({len(self.word2id)+1: "<pad>"})

        self.word2id.update({"<unk>": len(self.word2id)})
        self.word2id.update({"<pad>": len(self.word2id) + 1})


    def string2ids(self, content):
        res = []
        content = str(content).split(" ")
        for wd in content:
            if len(wd) == 0:
                continue
            else:
                res.append(self.word2id[wd])
        return res
    def ids2string(self):
        pass


class yelp_review_dataset(Dataset):
    def __init__(self, dataset_path, vocab):
        self.data = pd.read_csv(dataset_path)
        self.len = len(self.data)
        data_x_unpad = [vb.string2ids(txt) for txt in self.data['text']]
        data_x_pad = padding(data_x_unpad, 200, vb.word2id["<pad>"])
        self.data_x = torch.IntTensor(np.asarray(data_x_pad))
        self.data_y = torch.IntTensor(np.asarray([1 if label == 'pos' else 0 for label in self.data['sentiment']]))


    def __getitem__(self, index):
        return (self.data_x[index], self.data_y[index])

    def __len__(self):
        return self.len


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Embedding = nn.Embedding(105917, 128)
        self.LSTM = nn.LSTM(128, 256, 2,dropout=0)
        #self.dropout = nn.Dropout(0.3)
        #self.fc = nn.Linear(256, 2)
        self.fc = nn.Linear(256*2, 2)
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        embed = self.Embedding(input)
        #out = self.LSTM(embed)[0]
        state, hidden = self.LSTM(embed.permute([1, 0, 2]))
        out = torch.cat([state[0], state[-1]], dim=1)
        #out = self.fc(out)
        #out = self.dropout(out)
        out = self.fc(out)
        #sigout = self.sig(out)
        #return sigout[:,-1]
        #out = self.softmax(out)
        #return torch.argmax(out[:,-1], dim=1)

        #return out[:,-1]
        return out


def padding(data_unpad, max_len, pad_id):
    data_pad = []
    for content in data_unpad:
        if len(content) == max_len: data_pad.append(content)
        elif len(content) > max_len: data_pad.append(content[:max_len])
        else:
            pad_len = max_len - len(content)
            content.extend([pad_id] * pad_len)
            data_pad.append(content)
    return data_pad


batch_size = 32
viz = Visdom()
char_x = []
char_y = []
acc = [0]
vb = vocab("vocab.txt")
yelp_dataset = yelp_review_dataset("sares2.csv", vb)
train_dataset, test_dataset = random_split(yelp_dataset, [50000,10000])
yelp_review_train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
yelp_review_test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)
model = Model()
model.cuda()

criterion = nn.CrossEntropyLoss().to(0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def train(model, train_dataloader):
    model.train()
    train_loss = 0
    for i, data in enumerate(train_dataloader, 0):
        x = data[0].cuda()
        y = data[1].cuda()
        optimizer.zero_grad()
        output = model(x).type(torch.Tensor)
        y = y.type(torch.LongTensor)
        loss = criterion(output.cuda(), y.cuda())
        loss.backward()
        optimizer.step()
        train_loss+=loss
    loss_mean = train_loss / (i+1)
    return loss_mean


def test(model, test_dataloader, epoch, c_x, c_y, acc):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i ,data in enumerate(test_dataloader, 0):
            x = data[0].cuda()
            y = data[1].cuda()
            optimizer.zero_grad()
            output = model(x).type(torch.Tensor)
            y = y.type(torch.LongTensor)
            test_loss += criterion(output.cuda(), y.cuda()).item()
            correct += int((torch.argmax(output, dim=1) == y).sum())
        test_loss /= (i+1)
        c_x.append(epoch)
        c_y.append([test_loss, correct/len(test_dataset)])
        if correct/len(test_dataset) > max(acc):
            state = {'model':model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, "checkpoint1/senti_model_{}_{}.pt".format(epoch, correct/len(test_dataset)))
        acc.append(correct/len(test_dataset))
        return test_loss, correct


if __name__ == '__main__':
    for epoch in range(60):
        loss_mean = train(model, yelp_review_train_dataloader)
        print('Train Epoch: {}\t Loss: {:.4f}'.format(epoch, loss_mean.item()))
        test_loss, correct = test(model, yelp_review_test_dataloader, epoch, char_x, char_y, acc)
        viz.line(X=char_x, Y=char_y, win="training", opts=dict(title="loss&acc", showlegend=True, xlabel="epoch", ylabel="loss&acc", legend=["loss", "acc"]))
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_dataset), 100. * correct / len(test_dataset)))

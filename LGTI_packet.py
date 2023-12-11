import time

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CNN_MODULE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnn1 = nn.Conv1d(in_channels=in_channels, out_channels=200, kernel_size=5, stride=2)
        self.cnn2 = nn.Conv1d(200, 100, 4, 1)
        self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(100)
        self.pool = nn.AvgPool1d(2)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.pool(x)
        return self.flatten(x)


class Net(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, batch_size, out_features, num_heads, num_class,
                 dropout):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.bach_size = batch_size
        self.emb_ip = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.emb_tcp = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.emb_payload = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(batch_first=True, hidden_size=hidden_size, input_size=embedding_dim, bidirectional=True)
        self.cnn_p = nn.Sequential(nn.Conv1d(embedding_dim, 200, 5, 2), nn.BatchNorm1d(200), nn.Conv1d(200, 100, 5, 2),
                                   nn.BatchNorm1d(100), nn.MaxPool1d(7), nn.Flatten())
        self.cnn_lt = CNN_MODULE(in_channels=embedding_dim)
        self.cnn_li = CNN_MODULE(in_channels=embedding_dim)
        self.gh1 = Fusion(in_features1=200, in_features2=hidden_size * 2, out_features1=out_features,
                          num_heads=num_heads)
        self.gh2 = Fusion(in_features1=200, in_features2=hidden_size * 2, out_features1=out_features,
                          num_heads=num_heads)
        self.gp = Fusion(in_features1=500, in_features2=hidden_size * 2, out_features1=out_features,
                         num_heads=num_heads)
        self.line = nn.Linear(out_features * 3, num_class)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, Li, Lt, Lp):
        Li = self.emb_ip(Li)
        Lt = self.emb_tcp(Lt)
        Lp = self.emb_payload(Lp)
        packet = torch.cat([Li, Lt, Lp], -1)
        Li = Li.permute(0, 2, 1)
        Lt = Lt.permute(0, 2, 1)
        Lp = Lp.permute(0, 2, 1)
        cnn_Li = self.cnn_li(Li).squeeze(-1)
        cnn_Li = self.relu(cnn_Li)

        cnn_Lt = self.cnn_lt(Lt).squeeze(-1)
        cnn_Lt = self.relu(cnn_Lt)
        cnn_Lp = self.cnn_p(Lp).squeeze(-1)
        cnn_Lp = self.relu(cnn_Lp)

        global_feature = self.lstm_net(packet)

        g_li = self.gh1(cnn_Li, global_feature)

        g_lt = self.gh2(cnn_Lt, global_feature)

        g_p = self.gp(cnn_Lp, global_feature)

        features = torch.cat([g_p, g_lt, g_li], -1)
        line1 = self.drop(self.line(features))
        return line1

    def lstm_net(self, x):
        h0 = torch.zeros(2, x.shape[0], self.hidden_size).to(device)
        c0 = torch.zeros(2, x.shape[0], self.hidden_size).to(device)
        output, (h, o) = self.lstm(x, (h0, c0))
        hidden_feature = torch.cat([h[0], h[1]], -1)
        return hidden_feature


class Fusion(nn.Module):
    def __init__(self, in_features1, in_features2, out_features1, num_heads):
        super().__init__()
        self.trans = nn.Linear(in_features1, out_features1)
        self.q = nn.Linear(in_features=in_features1, out_features=out_features1)
        self.k = nn.Linear(in_features=in_features2, out_features=out_features1)
        self.v = nn.Linear(in_features=in_features2, out_features=out_features1)
        self.mhd = nn.MultiheadAttention(embed_dim=out_features1, num_heads=num_heads)
        self.liner = nn.Linear(out_features1, out_features1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features=out_features1)

    def forward(self, x1, x2):
        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x2)
        out, _ = self.mhd(q, k, v)
        out = self.bn(self.liner(out + self.trans(x1)))
        out = self.relu(out)
        return out


class LabelSmoothing(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Run_Net():
    def __init__(self, num_embeddings, embedding_dim, hidden_size, batch_size,
                 out_features, num_class, num_heads, lr, epoches, dropout, smoothing):
        super(Run_Net, self).__init__()
        self.net = Net(num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                       hidden_size=hidden_size, batch_size=batch_size, out_features=out_features, num_heads=num_heads,
                       num_class=num_class, dropout=dropout).to(device)
        self.batch_size = batch_size
        self.epoches = epoches
        self.loss_fun = LabelSmoothing(smoothing)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr, weight_decay=0.00001)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_class = num_class
        self.out_features = out_features
        self.num_heads = num_heads
        self.lr = lr

    def load_data(self, file):
        data = np.loadtxt(file, delimiter=' ')
        data = torch.tensor(data, dtype=torch.long)
        return data

    def dataloader(self, file):
        data = self.load_data(file)
        Li1 = data[:, 1:21]
        Lt1 = data[:, 21:41]
        payload1 = data[:, 41:193]

        labels = data[:, 0:1]
        labels = torch.tensor(labels, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(Li1.to(device), Lt1.to(device), payload1.to(device), labels.to(device))
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True,
                                                 drop_last=False)
        return dataloader

    def run(self, write_file, train_file, dev_file):
        save_f1 = 0
        f1 = open(write_file, 'a+')
        f1.write(
            "Run_Net(num_embeddings={}, embedding_dim={}, hidden_size={},batch_size={}, out_features1={}, num_class={}, drouout={}, num_heads={}, lr={}， epoches={})".format(
                self.num_embeddings, self.embedding_dim, self.hidden_size, self.batch_size, self.out_features,
                self.num_class, self.dropout, self.num_heads, self.lr, self.epoches) + '\n')
        train_dataloader = self.dataloader(train_file)
        trainlen = len(train_dataloader)
        dev_dataloader = self.dataloader(dev_file)
        devlen = len(dev_dataloader)
        for e in range(self.epoches):
            train_f1 = 0
            total_loss = 0
            train_precision = 0
            train_recall = 0
            train_acc = 0
            dev_f1 = 0
            dev_acc = 0
            dev_recall = 0
            dev_precision = 0
            print("Epoch:{}".format(e))
            f1.write("EPOCH " + str(e) + '\n')
            TOTAL_TRAIN = 0
            TOTAL_DEV = 0
            for i, (Li1, Lt1, payload1, label) in enumerate(train_dataloader):
                self.net.train()
                self.optimizer.zero_grad()
                train_start = time.time()
                out = self.net(Li1, Lt1, payload1)
                train_end = time.time()
                TOTAL_TRAIN += (train_end - train_start)
                loss = self.loss_fun(out, label.squeeze())
                total_loss = total_loss + loss.item()
                loss.backward()
                self.optimizer.step()
                pre = out.max(1)[1].cpu().detach().numpy()
                true = label.cpu().detach().numpy()
                train_f1 = train_f1 + f1_score(pre, true, average='macro')
                train_acc = train_acc + accuracy_score(pre, true)
                train_precision = train_precision + precision_score(pre, true, average='macro')
                train_recall = train_recall + recall_score(pre, true, average='macro')
            print("train_time:{:.4f}".format(TOTAL_TRAIN))
            print("train_acc:{:.4f}".format(train_acc / trainlen))
            print("train_f1:{:.4f}".format(train_f1 / trainlen))
            print("train_recall:{:.4f}".format(train_recall / trainlen))
            print("train_precision:{:.4f}".format(train_precision / trainlen))
            print("total_loss:{}".format(total_loss / trainlen))
            f1.write("train_acc:{:.4f}".format(train_acc / trainlen) + "\n")
            f1.write("train_f1:{:.4f}".format(train_f1 / trainlen) + "\n")
            f1.write("total_loss:{}".format(total_loss / trainlen) + "\n")
            with torch.no_grad():
                for i, (Li1, Lt1, payload1, label) in enumerate(dev_dataloader):
                    self.net.eval()
                    dev_start = time.time()
                    out = self.net(Li1, Lt1, payload1)
                    dev_end = time.time()
                    TOTAL_DEV += dev_end - dev_start
                    pre = out.max(1)[1].cpu().detach().numpy()
                    true = label.cpu().detach().numpy()
                    dev_f1 = dev_f1 + f1_score(pre, true, average='macro')
                    dev_acc = dev_acc + accuracy_score(pre, true)
                    dev_precision = dev_precision + precision_score(pre, true, average='macro')
                    dev_recall = dev_recall + recall_score(pre, true, average='macro')
            print("dev_time:{:.4f}".format(TOTAL_DEV))
            print("dev_acc:{:.4f}".format(dev_acc / devlen))
            print("dev_recall:{:.4f}".format(dev_recall / devlen))
            print("dev_f1:{:.4f}".format(dev_f1 / devlen))
            print("dev_precision:{:.4f}".format(dev_precision / devlen))
            if save_f1 < dev_f1:
                save_f1 = dev_f1
                print('save_model')
                torch.save(self.net, 'pth/ourmodel_packet.pth')
            f1.write("dev_acc:{:.4f}".format(dev_acc / devlen) + "\n")
            f1.write("dev_f1:{:.4f}".format(dev_f1 / devlen) + "\n")
            f1.write("dev_precision:{:.4f}".format(dev_precision / devlen) + "\n")
            f1.write("dev_recall:{:.4f}".format(dev_recall / devlen) + "\n")


if __name__ == '__main__':
    model = Run_Net(num_embeddings=256, embedding_dim=64, hidden_size=64, batch_size=128, out_features=128,
                    num_class=120, lr=0.001, epoches=100, num_heads=8, dropout=0.5, smoothing=0.05)

    model.run('result/LGTI_packet.txt', 'packetdata/data_train.txt', 'packetdata/data_dev.txt')

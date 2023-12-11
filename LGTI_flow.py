import torch.cuda

from LGTI_packet import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Run_Net():
    def __init__(self, num_embeddings, embedding_dim, hidden_size, batch_size,
                 out_features, num_class, num_heads, lr, epoches, dropout, smoothing):
        super(Run_Net, self).__init__()
        self.net = torch.load('pth/ourmodel_packet.pth').to(device)
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
        self.epoches = epoches

    def load_data(self, file):
        data = np.loadtxt(file, delimiter=' ')
        data = torch.tensor(data, dtype=torch.long)
        return data

    def dataloader(self, file):
        data = self.load_data(file)
        Li1 = data[:, 1:21]
        Lt1 = data[:, 21:41]
        payload1 = data[:, 41:193]

        Li2 = data[:, 194:214]
        Lt2 = data[:, 214:234]
        payload2 = data[:, 234:386]

        labels = data[:, 0:1]
        labels = torch.tensor(labels, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(Li1.to(device), Lt1.to(device), payload1.to(device),
                                                 Li2.to(device), Lt2.to(device), payload2.to(device),
                                                 labels.to(device))
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True,
                                                 drop_last=False)
        return dataloader

    def run(self, dev_file):
        dev_dataloader = self.dataloader(dev_file)
        devlen = len(dev_dataloader)
        for i, (Li1, Lt1, payload1, Li2, Lt2, payload2, label) in enumerate(dev_dataloader):
            self.net.eval()
            out1 = self.net(Li1, Lt1, payload1)
            out2 = self.net(Li2, Lt2, payload2)
            out = out1 + out2
            pre = out.max(1)[1].cpu().detach().numpy()
            true = label.cpu().detach().numpy()
            dev_f1 = dev_f1 + f1_score(pre, true, average='macro')
            dev_acc = dev_acc + accuracy_score(pre, true)
            dev_precision = dev_precision + precision_score(pre, true, average='macro')
            dev_recall = dev_recall + recall_score(pre, true, average='macro')
        print("dev_acc:{:.4f}".format(dev_acc / devlen))
        print("dev_recall:{:.4f}".format(dev_recall / devlen))
        print("dev_f1:{:.4f}".format(dev_f1 / devlen))
        print("dev_precision:{:.4f}".format(dev_precision / devlen))


if __name__ == '__main__':
    model = Run_Net(num_embeddings=256, embedding_dim=64, hidden_size=64, batch_size=128, out_features=128,
                    num_class=120, lr=0.001, epoches=100, num_heads=8, dropout=0.5, smoothing=0.05)
    model.run('flowdata/data_test.txt')

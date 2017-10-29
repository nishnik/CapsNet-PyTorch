import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

batch_size = 1
test_batch_size = 1
epochs = 10
lr = 0.01
momentum = 0.5
no_cuda = True
seed = 1
log_interval = 10

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

if cuda:
    torch.cuda.manual_seed(seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)

class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, 9) # First Conv
        conv_caps = [nn.Conv2d(256, 8, 9, stride = 2) for i in range(32)]
        self.conv_caps = nn.ModuleList(conv_caps) # Primary caps
        self.weight_matrices = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.Linear(8, 16) for i in range(6)]) for i in range(6)]) for i in range(32)]) # From primary caps to digit caps
        self.bij = Variable(torch.FloatTensor(32, 6, 6, 10).zero_()) # routing weights
    def forward(self, x):
        x = F.relu((self.conv1(x)))
        prim_caps_layer = [self.conv_caps[i](x).resize(8, 6, 6).permute(1, 2, 0) for i in range(32)]
        for k in range(len(prim_caps_layer)):
            for i in range(prim_caps_layer[k].size()[0]):
                for j in range(prim_caps_layer[k].size()[1]):
                    tmp = self.non_linearity(prim_caps_layer[k][i, j].clone())
                    prim_caps_layer[k][i, j] = tmp
        tmp = torch.stack(prim_caps_layer)
        out = Variable(torch.FloatTensor(32, 6, 6, 16))
        for i in range(32):
            for j in range(6):
                for k in range(6):
                    t = self.weight_matrices[i][j][k](tmp[i, j, k].clone())
                    out[i, j, k] = t
        # print (self.bij[0][0][0])
        for loop in range(10):
            si = Variable(torch.FloatTensor(10, 16).zero_())        
            for i in range(32):
                for j in range(6):
                    for k in range(6):
                        ci = F.softmax(self.bij[i,j,k].clone())
                        for m in range(10):
                            t = si[m].clone() + ci[m].clone() * out[i,j,k].clone()
                            si[m] = t
            for i in range(10):
                tmp = self.non_linearity(si[i].clone())
                si[i] = tmp
            for i in range(32):
                for j in range(6):
                    for k in range(6):
                        for m in range(10):
                            tmp = self.bij[i, j, k, m].clone() + si[m].dot(out[i,j,k].clone())
                            self.bij[i, j, k, m] = tmp
        # print (self.bij[0][0][0])
        norms = Variable(torch.FloatTensor(10))
        for i in range(10):
            norms[i] = si[i].norm()
        return norms, self.bij
    def non_linearity(self, vec):
        nm = vec.norm()
        nm2 = nm ** 2
        vec = vec * nm2 / ((1 + nm2) * nm)
        return vec

model = CapsNet()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
model.train()

point_nine = torch.FloatTensor(1)
point_nine.fill_(0.9)
point_nine = Variable(point_nine)

point_one = torch.FloatTensor(1)
point_one.fill_(0.1)
point_one = Variable(point_one)

point_five = torch.FloatTensor(1)
point_five.fill_(0.5)
point_five = Variable(point_five)

for batch_idx, (data, target) in enumerate(train_loader):
    data = Variable(data)
    target = target[0]
    optimizer.zero_grad()
    output = model(data)
    norms = output[0]
total_loss = 0
for i in range(10):
    print (i)
    if (i == target):
        loss = torch.max(Variable(torch.zeros(1)), point_nine - norms[i])
        loss.backward(retain_graph = True)
        optimizer.step()
        total_loss += loss
        print (loss)
    else:
        loss = point_five * torch.max(Variable(torch.zeros(1)), norms[i] - point_one)
        loss.backward(retain_graph = True)
        optimizer.step()
        print (loss)


import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class OneLayerNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(OneLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self ,x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(ThreeLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1,H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self ,x):
        h1_relu = self.linear1(x).clamp(min=0)
        h2_relu = self.linear2(h1_relu).clamp(min=0)
        y_pred = self.linear3(h2_relu)
        return y_pred


class FourLayerNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(FourLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1,H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, D_out)

    def forward(self ,x):
        h1_relu = self.linear1(x).clamp(min=0)
        h2_relu = self.linear2(h1_relu).clamp(min=0)
        h3_relu = self.linear3(h2_relu).clamp(min=0)
        y_pred = self.linear4(h3_relu)
        return y_pred





#N, D_in, H, D_out = 64, 1000, 100, 10

inp = np.load('train.npy')
out = np.load('train_gt.npy')

x = torch.from_numpy(inp)
y = torch.from_numpy(out)

x = x.type(torch.FloatTensor)
y = y.type(torch.FloatTensor)

#x = torch.randn(N, D_in)
#y = torch.randn(N, D_out)

N = x.shape[0]
D_in = x.shape[1]
D_out = 1
H = 50
H1 = 50
H2 = 50
H3 = 50

y = y.view(N, D_out)

print("N = " ,N, " D_in = ", D_in , " D_out = ", D_out, " H = ", H)

loader = DataLoader(TensorDataset(x, y), batch_size=50)

#model = OneLayerNet(D_in, D_out)
#model = TwoLayerNet(D_in, H, D_out)
#model = ThreeLayerNet(D_in, H1, H2, D_out)
model = FourLayerNet(D_in, H1, H2, H3, D_out)

optimizer = torch.optim.RMSprop(model.parameters(), lr = 1e-2)
"""for t in range(5000):
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)

    print(t, loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
"""
f = open("fourlayerlr-2.txt","w")

for epoch in range(501):
    i = 0
    sumloss = 0
    for x_batch, y_batch in loader:
        y_pred = model(x_batch)

        loss = torch.nn.functional.mse_loss(y_pred, y_batch, size_average=False)
        sumloss += loss.item()
        #print(epoch,i, loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        i += 1
    sumloss /= N
    print(epoch+1, sumloss)
    f.write(str(epoch+1) + " " + str(sumloss) + "\n")
    if(epoch % 10 == 0):
        torch.save(model,"model_"+str(epoch)+".pt")

f.close()

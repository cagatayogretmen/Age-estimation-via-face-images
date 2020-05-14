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



inp = np.load('benler.npy')

x = torch.from_numpy(inp)

x = x.type(torch.FloatTensor)

N = x.shape[0]
D_in = x.shape[1]
D_out = 1

model = torch.load("testmodel.pt")

y_pred = model(x)
yaz = y_pred.detach().numpy()
yaz = yaz.flatten()

np.save("benler_estimation.npy", yaz)

print(yaz)

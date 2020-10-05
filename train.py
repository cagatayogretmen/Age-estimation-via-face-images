#input ve output train etme

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np



class OneLayerNet(torch.nn.Module):
    #d_in:imgler(toplam boyutun girişi çıkış: bout 1boyutunda dönüşüm yapıyor:512 numpy array veri giriyor 1 çıkıyor)
    def __init__(self, D_in, D_out):
        super(OneLayerNet, self).__init__()
        #katman oluşturuyor.linear transformation uyguluyor
        self.linear1 = torch.nn.Linear(D_in, D_out)

    #forward ediyor
    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred


class TwoLayerNet(torch.nn.Module):
    #h:katmanlar arasında 1.katman sonunda kullanılacak boyut 2.katmanın girişi için e kullanılan boyut
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self ,x):
        #0 den küçük değerleri küçük yap
        #ReLu Fonksiyonu f (x) = max (0, x) 
        h_relu = self.linear1(x).clamp(min=0)
        #katman sonunda aktivasyona giriyor oluşan sonraki katmana veriliyor eğer sona geldiyse y_pred döndürür
        y_pred = self.linear2(h_relu)
        return y_pred

#neden relu
#ReLU [0, +∞) 
#çok fazla nironlu ağda düşünelim tanh ve sigmoid tüm nöronları işlkeme sokar
#relu ile bazı nöronlar aktif olup verimli hesaplama yaparız. negatif eksende 0 alması daha hızlı olmasını sağlar
#leaky relu ile pozitiflerin de türevi alınır

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

#relu:giriş değeri 0 altındaysa çıktı 0, 0 üstündeyse giriş değeri çıkış değeridir.bağımlı değişkenle doğrusal ilişkidedir.




#N, D_in, H, D_out = 64, 1000, 100, 10

inp = np.load('datasets/train.npy')
out = np.load('datasets/train_gt.npy')

#numpy vektörünü torch a çevirdik
x = torch.from_numpy(inp)
y = torch.from_numpy(out)

#type değiştirir
#katmanların weight ile çarpılabilmesi için float cinsine çeviririr
x = x.type(torch.FloatTensor)
y = y.type(torch.FloatTensor)

#x = torch.randn(N, D_in)
#y = torch.randn(N, D_out)

#x.shape: 5000,512
N = x.shape[0] #resim sayısı:5000
D_in = x.shape[1] #resim boyutu:512
D_out = 1 #resmin label i
H = 50 #denendi
H1 = 50
H2 = 50
H3 = 50


y = y.view(N, D_out)

print("N = " ,N, " D_in = ", D_in , " D_out = ", D_out, " H = ", H)

#x ve y yi 50 lik parçalara böler kendi içinde kaydeder
#çünkü tüm veriyi daha fazla forward backward olur .her epoch 500x100 defa matriksleri güncelliyor 
loader = DataLoader(TensorDataset(x, y), batch_size=50)

#model = OneLayerNet(D_in, D_out)
#model = TwoLayerNet(D_in, H, D_out)
#model = ThreeLayerNet(D_in, H1, H2, D_out)
model = FourLayerNet(D_in, H1, H2, H3, D_out)

#rmsprop kullandık. matriks değerlerini güncelliyor
#çıkan hatalara göre yeni ağırlık belirlenmesi
#rmsprop gradient descent yapısındadır
#rmsprop aşırı küçültme sorununu çözer, tüm eğilimlerin karelerinden elde etmek yerine belirli çevreçe boyutu ile kısılamıştır

#relu ak. fonksiyonu ile rmsprop un daha yüksek doğrulukda çalıştığı gözlenmeştir
#aktv.:tanh, relu
#optimizer:sgd, adagrad, nadam, adam, rmsprop
#adam optimizasyon yönteminin hızlı olduğu fakat reluyla rmspropun daha hızlı olduğu saptanmıştır
#hızı aktivasyon fonksiyonu, optimizer, epoch sayısı
optimizer = torch.optim.RMSprop(model.parameters(), lr = 1e-2) #learning rate belirttik
'''
makine öğrenimindeki öğrenme hızı veya adım büyüklüğü olarak adlandırılır.
learning rate büyük olursa minimum olan nokraya ulaşılamaz, çok küçük olursa da model çok yavaş öğrenir.
* deneme yapılarak en optimum learning rate verilmesi tavsiye edilir.
'''

"""for t in range(5000):
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)

    print(t, loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
"""
#learning rate 10**2 lisi.denendi en iyisi
f = open("fourlayerlr-2.txt","w")

for epoch in range(501):
    i = 0
    sumloss = 0
    for x_batch, y_batch in loader:
        y_pred = model(x_batch)

        loss = torch.nn.functional.mse_loss(y_pred, y_batch, size_average=False)
        sumloss += loss.item()
        #print(epoch,i, loss.item())

        #gradient.loss b. gradientleri.peşpeşe kullanılmak zorunda.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        i += 1
    sumloss /= N
    print(epoch+1, sumloss)
    f.write(str(epoch+1) + " " + str(sumloss) + "\n")
    if(epoch % 10 == 0):
        torch.save(model,"models/model_"+str(epoch)+".pt")

f.close()

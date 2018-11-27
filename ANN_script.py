import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import rnn
import torch.utils.data as Data
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import preprocessing
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
a = np.load('/home/lihao/GDB/GDB_data/inputs_only7.npy')
b = np.load('/home/lihao/GDB/GDB_data/aes_only7.npy')
x_train,x_test,y_train,y_test = train_test_split(a,b,test_size=0.1)

x_train = torch.from_numpy(x_train).type(torch.DoubleTensor)
y_train = torch.from_numpy(y_train).type(torch.DoubleTensor)
y_train = torch.unsqueeze(y_train,dim=1)
x_train,y_train = Variable(x_train),Variable(y_train)

x_test = torch.from_numpy(x_test).type(torch.DoubleTensor)
y_test = torch.from_numpy(y_test).type(torch.DoubleTensor)
y_test = torch.unsqueeze(y_test,dim=1)
x_test,y_test = Variable(x_test),Variable(y_test)

class TwistingTanh(torch.autograd.Function):
    '''
    this function is recommended for general purposes in deep learning 
    f(x) = 1.7159*tanh(2x/3)+ax; while a = 1
    '''
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return 1.7159 + torch.tanh(input * 2 / 3) + input
    
    @staticmethod
    def backward(ctx,grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (2.1438 - 1.1438 * (torch.tanh(2 * input / 3)) ** 2)
        return grad_input

FUNC = TwistingTanh.apply
    
class Net(torch.nn.Module):        
    def __init__(self,n_feature, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_output):
        super(Net,self).__init__()
        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden_1)
        self.hidden_2 = torch.nn.Linear(n_hidden_1,n_hidden_2)
        self.hidden_3 = torch.nn.Linear(n_hidden_2,n_hidden_3)
        self.hidden_4 = torch.nn.Linear(n_hidden_3,n_hidden_4)
        self.predict = torch.nn.Linear(n_hidden_4, n_output)
    def forward(self,x):
        x = FUNC(self.hidden_1(x))
        x = FUNC(self.hidden_2(x))
        x = FUNC(self.hidden_3(x))
        x = FUNC(self.hidden_4(x))
        x = self.predict(x)
        return x

epoch = 200
BATCH_SIZE = 256
torch_dataset = Data.TensorDataset(x_train,y_train)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)  
N = 8
net = Net(276,N,N,N,N,1)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.001,betas=(0.9,0.99))
net = net.double()
net.cuda()

dat = []
for epoch in range(epoch):
    for step, (batch_x,batch_y) in enumerate(loader):        
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        prediction = net(batch_x)
        loss = loss_func(prediction,batch_y)
        dat.append(loss.cpu())
        print('epoch:',epoch,'step:',step,'MSEloss',loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
plt.plot(dat)
plt.ylim((0,0.02))
x_test,y_test = x_test.cuda(),y_test.cuda()
prediction_test = net(x_test)
loss_test = loss_func(prediction_test,y_test)  
print('test_loss',loss_test.cpu().detach().numpy())        
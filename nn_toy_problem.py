import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import abc as container_abcs
from torchvision import datasets, transforms
import numpy as np
import torch.utils as utils
import matplotlib.pyplot as plt
import scipy.special as spc
# from torch._six import container_abcs

def vibrating_drum(arr):
    # terrible implementation
    m = arr[:,0]
    n = arr[:,1]
    r = arr[:,2]
    th = arr[:,3]
    y = np.zeros(len(m))
    for k in range(len(m)):
        y[k] = spc.jv(m[k], spc.jn_zeros(m[k], n[k])[-1] * r[k]) * np.cos(m[k] * th[k])
    return y


n = 25 # this is the number of data used
d = 1 # this is the dimension of Xi

X = np.array([-2, -1, 0, 1, 2])
y = np.array([1, -1, 1, 1, -1])
# y = np.random.rand(n)
# def


train_x = torch.Tensor(X) 
train_y = torch.Tensor(y)

dataset = utils.data.TensorDataset(train_x, train_y)
dataloader = utils.data.DataLoader(dataset)

n_input = d
n_hidden_v = np.array([1, 5, 10, 25, 100, 200])
# n_hidden_v = [15]
n_output = 1
epochs = 5000

loss_at_epoch = np.zeros((epochs, len(n_hidden_v)))

for n_hidden in n_hidden_v:

    model = nn.Sequential(nn.Linear(n_input, n_hidden),
                          nn.ReLU(),
                          nn.Linear(n_hidden, n_output))
    # Define the loss
    # reg = lambda a : a + 10
    criterion = nn.MSELoss(reduction='sum')
    
    # Optimizers require the parameters to optimize and a learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-5)
    
    
    for e in range(epochs):
        
        running_loss = 0
        
        for train_x, train_y in dataloader:
            train_x = train_x.view(train_x.shape[0], -1)
            optimizer.zero_grad()
            output = model(train_x)
            loss = criterion(output, train_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        else:
            # print(running_loss/len(dataloader)) # average loss over the epoch
            # print(np.where(n_hidden == n_hidden_v))
            loss_at_epoch[e, np.where(n_hidden_v == n_hidden)[0]] = running_loss/len(dataloader)
            

#plotting
plt.figure()
for n_hidden in n_hidden_v:
    plt.plot(range(epochs), loss_at_epoch[:, np.where(n_hidden_v == n_hidden)[0]], label = "hidden nodes = " + str(n_hidden))

plt.yscale('log')
plt.legend()
plt.xlabel("num epochs")
plt.ylabel("MSE Loss")
# https://towardsdatascience.com/training-a-neural-network-using-pytorch-72ab708da210
#https://stackoverflow.com/questions/57949625/with-pytorch-dataloader-how-to-take-in-two-ndarray-data-label

### CHANGE THE LEARNING RATE














    
    
    
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
import random


# from torch._six import container_abcs

def my_loss(output, target, model, beta):
    
    loss = torch.norm((output - target)**2)/2 
    
    for layer, p in enumerate(model.parameters()):

        if layer == 0:
            loss = loss + beta/2*torch.norm(p)**2
        if layer == 1:
            
            loss = loss + beta/2 * sum([torch.norm(p[j], 1)**2 for j in range(p.shape[0])])
    return loss


beta = 1e-4
np.random.seed(10)
n = 10
d = 3
m = 4
X = np.random.randn(n,d-1)
X = np.append(X,np.ones((n,1)), axis=1)
y=((np.linalg.norm(X[:,0:d-1],axis=1)>1)-0.5)*2

train_x = torch.Tensor(X) 
train_y = torch.Tensor(y)

dataset = utils.data.TensorDataset(train_x, train_y)
dataloader = utils.data.DataLoader(dataset)

n_input = d
n_hidden_v = np.array([m, 5, 10, 100])

n_output = d
epochs = 5000

loss_at_epoch = np.zeros((epochs, len(n_hidden_v)))

for n_hidden in n_hidden_v:

    model = nn.Sequential(nn.Linear(n_input, n_hidden),
                          nn.ReLU(),
                          nn.Linear(n_hidden, n_output))
    # Optimizers require the parameters to optimize and a learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    
    
    for e in range(epochs):
        
        running_loss = 0
        
        for train_x, train_y in dataloader:
            train_x = train_x.view(train_x.shape[0], -1)
            optimizer.zero_grad()
            output = model(train_x)
            loss = my_loss(output, train_y, model, beta)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:

            loss_at_epoch[e, np.where(n_hidden_v == n_hidden)[0]] = running_loss/len(dataloader)
            

#plotting
plt.figure()
for n_hidden in n_hidden_v:
    plt.plot(range(epochs), loss_at_epoch[:, np.where(n_hidden_v == n_hidden)[0]], label = "hidden nodes = " + str(n_hidden))

plt.yscale('log')
plt.legend()
plt.xlabel("num epochs")
plt.ylabel("MSE Loss")
print(loss_at_epoch[4999])
# https://towardsdatascience.com/training-a-neural-network-using-pytorch-72ab708da210
#https://stackoverflow.com/questions/57949625/with-pytorch-dataloader-how-to-take-in-two-ndarray-data-label

### CHANGE THE LEARNING RATE














    
    
    
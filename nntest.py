import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import torch.utils as utils
n = 50 # this is the number of data used
d = 5 # this is the dimension of Xi
X = np.random.rand(n,d) 
y = np.random.rand(n)


train_x = torch.Tensor(X)
train_y = torch.Tensor(y)

dataset = utils.data.TensorDataset(train_x, train_y)
dataloader = utils.data.DataLoader(dataset)

n_input = d
n_hidden = 100
n_output = d

model = nn.Sequential(nn.Linear(n_input, n_hidden),
                      nn.ReLU(),
                      nn.Linear(n_hidden, n_output),
                      nn.LogSoftmax(dim=1))
# Define the loss
criterion = nn.MSELoss(reduction='sum')

# Optimizers require the parameters to optimize and a learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



epochs = 10000
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
        print(running_loss/len(dataloader))


# https://towardsdatascience.com/training-a-neural-network-using-pytorch-72ab708da210
#https://stackoverflow.com/questions/57949625/with-pytorch-dataloader-how-to-take-in-two-ndarray-data-label
















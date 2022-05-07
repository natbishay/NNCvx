from mpl_toolkits import mplot3d
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



r = np.linspace(0.01, 1, 25)
th = np.linspace(0, 2*np.pi, 36)
# X = np.stack((x1, x2), axis = 1) 
Xdr = np.array(np.meshgrid([0,1,2], [1,2], r, th)).T.reshape(-1,4)
ydr = vibrating_drum(Xdr) 
ydr = ydr + np.random.normal(0, 0.1, len(ydr))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x1 = Xdr[:,2] * np.cos(Xdr[:,3])
x2 = Xdr[:,2] * np.sin(Xdr[:,3])

eval_x_np =np.array(np.meshgrid([2], [1], r, th )).T.reshape(-1,4)
eval_x = torch.tensor(eval_x_np)
eval_y_exact = vibrating_drum(eval_x_np)
train_x = torch.Tensor(Xdr)
train_y = torch.Tensor(ydr)

dataset = utils.data.TensorDataset(train_x, train_y)
dataloader = utils.data.DataLoader(dataset)


n = np.size(Xdr, axis=0)# this is the number of data used
d = np.size(Xdr, axis=1) # this is the dimension of Xi
n_input = d
# n_hidden_v = np.array([1, 5, 10, 25, 100, 200])

n_hidden_v = [15, 50]
n_output = 1
epochs = 50

loss_at_epoch = np.zeros((epochs, len(n_hidden_v)))

for j in range(len(n_hidden_v)):
    n_hidden = n_hidden_v[j]
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
            print(running_loss/len(dataloader)) # average loss over the epoch
            # print(np.where(n_hidden == n_hidden_v))
            loss_at_epoch[e, j] = running_loss/len(dataloader)
            

#plotting (not good code)
plt.figure()
for j in range(len(n_hidden_v)):
    n_hidden = n_hidden_v[j]
    plt.plot(range(epochs), loss_at_epoch[:, j], label = "hidden nodes = " + str(n_hidden))

plt.yscale('log')
plt.legend()
plt.xlabel("num epochs")
plt.ylabel("MSE Loss")

# eval
y_eval = model(eval_x.float())
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X,Y = (eval_x_np[:,2] * np.cos(eval_x_np[:,3]), 
           eval_x_np[:,2] * np.sin(eval_x_np[:,3]))
ax.plot_trisurf(X, Y, y_eval.detach().numpy()[:,0])
ax.plot_trisurf(X, Y, eval_y_exact, cmap='binary')

# https://towardsdatascience.com/training-a-neural-network-using-pytorch-72ab708da210
#https://stackoverflow.com/questions/57949625/with-pytorch-dataloader-how-to-take-in-two-ndarray-data-label
















    
    
    
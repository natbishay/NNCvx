import torch
from torch import nn
import torch.nn.utils.prune as prune
import numpy as np
import torch.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from math import isclose
import matplotlib.pyplot as plt




global beta
global n
global d
global lr
global X
global y
global alpha
beta = 1e-4
epochs = 10
n = 10
d = 3 # dimension of input
lr=0.005
np.random.seed(10)
X = np.append(np.random.randn(n,d-1),np.ones((n,1)), axis=1)
y=((np.linalg.norm(X[:,0:d-1],axis=1)>1)-0.5)*2


def alpha_finder():
    def drelu(x):
        return x>=0
    losses = []
    num_hidden_neurons = []
    D=np.empty((n,0))
    ## Finite approximation of all possible sign patterns
    for i in range(int(1e2)):
        u=np.random.randn(d,1)
        D=np.append(D,drelu(np.dot(X,u)),axis=1)
    D=(np.unique(D,axis=1))
    m = D.shape[1]
    v = cp.Variable((d, m)) # d by m
    w = cp.Variable((d, m))
    constraints = []
    obj_a = cp.sum(cp.multiply(D , (X@(v - w))), axis=1)
    cost=cp.sum(cp.pos(1-cp.multiply(y,obj_a)))/n + beta*(cp.mixed_norm(v.T,2,1)+cp.mixed_norm(w.T,2,1))
    constraints += [cp.multiply(2*D - np.ones((n,m)), (X@v)) >= 0]
    constraints += [cp.multiply(2*D - np.ones((n,m)), (X@w)) >= 0]
    obj_val = cp.Minimize(cost)
    prob = cp.Problem(obj_val, constraints)
    prob.solve()
    u = np.zeros((d,m))
    alpha = np.zeros((m))
    for i in range(m):
        norm_vi = np.linalg.norm(v.value[:,i])
        norm_wi = np.linalg.norm(w.value[:,i])
        if isclose(norm_vi,0,abs_tol=1e-4):
            if isclose(norm_wi,0,abs_tol=1e-4):
                u[:,i] = 0
            else:
                u[:,i] = w.value[:,i]/np.sqrt(norm_wi)
                alpha[i] = np.sqrt(norm_wi)
        else:
            u[:,i] = v.value[:,i]/np.sqrt(norm_vi)
            alpha[i] = np.sqrt(norm_vi)
            
    return alpha

def my_loss(output, target, model, beta):
    loss = torch.norm((output - target)**2)/2 
    for layer, p in enumerate(model.parameters()):
        if layer == 0:
            loss = loss + beta/2*torch.norm(p)**2
        if layer == 1:   
            loss = loss + beta/2 * sum([torch.norm(p[j], 1)**2 for j in range(p.shape[0])])
    return loss

# first layer u pruning
def foobar_unstructured(module,name):
    FooBarPruningMethod.apply(module, name)
    return module


class FooBarPruningMethod(prune.BasePruningMethod):

    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        index = np.where(alpha == 0)[0]
        if mask.size()[0] > mask.size()[1]:
            mask.view(mask.size())[index,:] = 0
        else:
            mask.view(mask.size())[:,index] = 0
        return mask


alpha = alpha_finder()
m = np.size(alpha)# number of hidder nodes
train_x = torch.Tensor(X) 
train_y = torch.Tensor(y)

dataset = utils.data.TensorDataset(train_x, train_y)
dataloader = utils.data.DataLoader(dataset)

n_input = d
n_hidden_v = np.array([m])

n_output = d


loss_at_epoch = np.zeros((epochs, len(n_hidden_v)))

for n_hidden in n_hidden_v:

    model = nn.Sequential(nn.Linear(n_input, n_hidden),
                          nn.ReLU(),
                          nn.Linear(n_hidden, n_output))


    # Optimizers require the parameters to optimize and a learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr)
    
    
    for e in range(epochs):
        
        running_loss = 0
        
        for train_x, train_y in dataloader:
            train_x = train_x.view(train_x.shape[0], -1)
            
            # apply masking to the model
            foobar_unstructured(model[0], name='weight')
            foobar_unstructured(model[2], name='weight')
            
            # permanent pruning
            prune.remove(model[0], 'weight')
            prune.remove(model[2], 'weight')
          #  print(model[0].weight)
          #  print(model[2].weight)
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
plt.ylabel("MSE Loss Pruning")
print(loss_at_epoch[4999])
# https://towardsdatascience.com/training-a-neural-network-using-pytorch-72ab708da210
#https://stackoverflow.com/questions/57949625/with-pytorch-dataloader-how-to-take-in-two-ndarray-data-label

### CHANGE THE LEARNING RATE














    
    
    
import numpy as np
import cvxpy as cp
from math import isclose
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.utils as utils

np.random.seed(10)
n = 10 # number of data
d = 3 # dimension of individual data
X = np.random.randn(n,d-1)
X = np.append(X,np.ones((n,1)), axis=1)
y=((np.linalg.norm(X[:,0:d-1],axis=1)>1)-0.5)*2

beta_opt = 1e-4
betas = [10**(j-10) for j in range(10)]
#betas = betas[0]

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

#obj_a = cp.reshape(cp.sum(cp.multiply(D , (X@(v - w))), axis=1), (n,d))
obj_a = cp.sum(cp.multiply(D , (X@(v - w))), axis=1)
cost=cp.sum(cp.pos(1-cp.multiply(y,obj_a)))/n + beta_opt*(cp.mixed_norm(v.T,2,1)+cp.mixed_norm(w.T,2,1))


constraints += [cp.multiply(2*D - np.ones((n,m)), (X@v)) >= 0]
constraints += [cp.multiply(2*D - np.ones((n,m)), (X@w)) >= 0]

obj_val = cp.Minimize(cost)

prob = cp.Problem(obj_val, constraints)
prob.solve()
# print("BETA =", beta)
# print(prob.value)
losses.append(prob.value)
# print(v.value)
# print(w.value)

# u = np.zeros((d,m))
# alpha = np.zeros((m))

# for i in range(m):
#     norm_vi = np.linalg.norm(v.value[:,i])
#     norm_wi = np.linalg.norm(w.value[:,i])
#     if isclose(norm_vi,0,abs_tol=1e-4):
#         if isclose(norm_wi,0,abs_tol=1e-4):
#             u[:,i] = 0
#         else:
#             u[:,i] = w.value[:,i]/np.sqrt(norm_wi)
#             alpha[i] = np.sqrt(norm_wi)
#     else:
#         u[:,i] = v.value[:,i]/np.sqrt(norm_vi)
#         alpha[i] = np.sqrt(norm_vi)

# num_hidden_neurons.append(np.nonzero(alpha)[0].shape[0])

# take the output of the optimization as input of NN

X = np.multiply(D , (X@(v.value - w.value))) 
epochs = 5000
n , d = np.shape(X)
lr=0.05
betas = [10**(j-10) for j in range(10)] # YOU CAN ALSO CHANGE THE BETAS
#y=((np.linalg.norm(X[:,0:d-1],axis=1)>1)-0.5)*2
#betas = [1e-9, 1e-8, 1e-7, 1e-4]
iter_ind = 0
loss_at_epoch = np.zeros((epochs, len(betas)))
for betai in betas:
    global beta
    beta = betai
    
    def alpha_finder():
        def drelu(x):
            return x>=0
        losses = []
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
    
    
    
    global alpha
    alpha = alpha_finder()
    m = np.size(alpha) # number of hidder nodes
    zeros = len(np.where(alpha == 0)[0]) # number of nonzeros
    train_x = torch.Tensor(X) 
    train_y = torch.Tensor(y)

    dataset = utils.data.TensorDataset(train_x, train_y)
    dataloader = utils.data.DataLoader(dataset)
    n_input = d
    n_hidden = [m, 5, 10, 20, 100]
    loss_at_epoch_nodes = np.zeros((epochs, len(n_hidden)))
    n_output = 1
    # model = nn.Sequential(
    #     nn.Linear(n_input, n_hidden),
    #     nn.ReLU(),
    #     nn.Linear(n_hidden, n_output),

        
    # )



    # Optimizers require the parameters to optimize and a learning rate
    
    for num_i, num_hid in enumerate(n_hidden):


        model = nn.Sequential(
        nn.Linear(n_input, num_hid),
        nn.ReLU(),
        nn.Linear(num_hid, n_output),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr)

        
    
        for e in range(epochs):
            
            running_loss = 0
            
            for train_x, train_y in dataloader:
                train_x = train_x.view(train_x.shape[0], -1)
                
                # apply masking to the model, for the m case
                if num_i == 0: # first one, which is the one we want to mask
                    foobar_unstructured(model[0], name='weight')
                    foobar_unstructured(model[2], name='weight')
                    #foobar_unstructured(model[4], name='weight')
                    
                    # permanent pruning
                    prune.remove(model[0], 'weight')
                    prune.remove(model[2], 'weight')
                #prune.remove(model[4], 'weight')
            # #    print(model[0].weight)
            #  print(model[2].weight)
                optimizer.zero_grad()
                output = model(train_x)
                loss = my_loss(output, train_y, model, beta)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            else:
                loss_at_epoch_nodes[e, num_i] = running_loss/len(dataloader)
                #loss_at_epoch[e, iter_ind] = running_loss/len(dataloader)
        iter_ind = iter_ind + 1   

#plotting
    plt.figure()
    for i in range(len(n_hidden)):
        #plt.plot(range(epochs), loss_at_epoch[:, i], label = "beta = " + str(betas[i]))
        if i == 0:
            plt.plot(loss_at_epoch_nodes[:,i], label="NH = "+str(n_hidden[i]) + ", not pruned = "+str(n_hidden[i]-zeros))
            continue
        plt.plot(loss_at_epoch_nodes[:,i], label="Num hidden = "+str(n_hidden[i]))

    plt.yscale('log')
    plt.legend()
    plt.xlabel("num epochs")
    plt.ylabel("MSE Loss Pruning")
    plt.title("Beta value: "+str(beta))
    print(loss_at_epoch_nodes[epochs-1])
    plt.savefig("final_nodes_m_pruned_"+str(beta)+".png")


plt.figure()
plt.plot(betas,loss_at_epoch[-1, :])
#plt.semilogx()
#plt.semilogy()
plt.xlabel("beta")
plt.ylabel("MSE final Loss Pruning")
# https://towardsdatascience.com/training-a-neural-network-using-pytorch-72ab708da210
#https://stackoverflow.com/questions/57949625/with-pytorch-dataloader-how-to-take-in-two-ndarray-data-label

### CHANGE THE LEARNING RATE








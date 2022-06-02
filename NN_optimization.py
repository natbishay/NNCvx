import torch
from torch import nn
import torch.nn.utils.prune as prune
import numpy as np
import torch.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from math import isclose
from collections import OrderedDict
import matplotlib.pyplot as plt
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor


def drelu(x):
    return x>=0

global n
global d
global lr
global X
global y

####################################################
# THE PARAMETERS SHOULD MATCH THE PREVIOUS CASES
##################################################
epochs = 100
n = 10
d = 3 # dimension of input
lr=0.05
betas = [10**(j-10) for j in range(10)] # YOU CAN ALSO CHANGE THE BETAS
################################################
np.random.seed(10)
X = np.append(np.random.randn(n,d-1),np.ones((n,1)), axis=1)
y=((np.linalg.norm(X[:,0:d-1],axis=1)>1)-0.5)*2


iter_ind = 0
loss_at_epoch = np.zeros((epochs, len(betas)))
losses_opt = []
for betai in betas:
    global beta
    beta = betai
    
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
    m = np.size(alpha)# number of hidder node
    train_x = torch.Tensor(X) 
    train_y = torch.Tensor(y)

    dataset = utils.data.TensorDataset(train_x, train_y)
    dataloader = utils.data.DataLoader(dataset)
    #m = d
    n_input = d
    n_hidden = m
    n_output = 1
    
    model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(n_input, n_hidden)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(n_hidden, 1)),
    # ('relu2', nn.ReLU()),
    # ('fc3', nn.Linear(n_hidden, n_output)),
]))



    # Optimizers require the parameters to optimize and a learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr)
    
    
    for e in range(epochs):
        
        running_loss = 0
        
        for train_x, train_y in dataloader:
            train_x = train_x.view(train_x.shape[0], -1)
            
            # apply masking to the model
            foobar_unstructured(model.fc1, name='weight')
            foobar_unstructured(model.fc2, name='weight')
            
            # permanent pruning
            prune.remove(model.fc1, 'weight')
            prune.remove(model.fc2, 'weight')
        #    print(model[0].weight)
          #  print(model[2].weight)
            optimizer.zero_grad()
            output = model(train_x)
            loss = my_loss(output, train_y, model, beta)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:

            loss_at_epoch[e, iter_ind] = running_loss/len(dataloader)
    iter_ind = iter_ind + 1   
    
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.fc1.register_forward_hook(get_activation('fc1'))
    
    
    x_second = []
    for k in range(n):
        train_x_k = torch.Tensor(X[k,:]) 
        output = model(train_x_k)
        x_second.append(activation['fc1'].numpy())
        
   # y_second=((np.linalg.norm(x_second[:,0:d-1],axis=1)>1)-0.5)*2
    y_second = y
    x_second = np.vstack(x_second)
    # the set of x that produces the most possible y
   # x_second = model(train_x).detach().numpy()
 # solve the second two layers using convex optimization  
    n_opt, d_opt = np.shape(x_second)
    D_opt=np.empty((n_opt,0))

    ## Finite approximation of all possible sign patterns
    for i in range(int(1e2)):
        u_opt=np.random.randn(d_opt,1)
        D_opt=np.append(D_opt,drelu(np.dot(x_second,u_opt)),axis=1)

    D_opt=(np.unique(D_opt,axis=1))
    m_opt = D_opt.shape[1]

    v_opt = cp.Variable((d_opt, m_opt)) # d by m
    w_opt = cp.Variable((d_opt, m_opt))

    constraints_opt = []

    #obj_a = cp.reshape(cp.sum(cp.multiply(D , (X@(v - w))), axis=1), (n,d))
    obj_a_opt = cp.sum(cp.multiply(D_opt , (x_second@(v_opt - w_opt))), axis=1)
    cost_opt=cp.sum(cp.pos(1-cp.multiply(y_second,obj_a_opt)))/n_opt + beta*(cp.mixed_norm(v_opt.T,2,1)+cp.mixed_norm(w_opt.T,2,1))


    constraints_opt += [cp.multiply(2*D_opt - np.ones((n_opt,m_opt)), (x_second@v_opt)) >= 0]
    constraints_opt += [cp.multiply(2*D_opt - np.ones((n_opt,m_opt)), (x_second@w_opt)) >= 0]

    obj_val_opt = cp.Minimize(cost_opt)

    prob_opt = cp.Problem(obj_val_opt, constraints_opt)
    prob_opt.solve()
    # print("BETA =", beta)
    # print(prob.value)
    losses_opt.append(prob_opt.value)
#plotting
plt.figure()
plt.plot(betas,losses_opt)
plt.semilogx()
plt.semilogy()
plt.xlabel("beta")
plt.ylabel("MSE final Loss Pruning")
#for i in range(iter_ind):
#    plt.plot(range(epochs), losses_opt[:, i], label = "beta = " + str(betas[i]))

# plt.yscale('log')
# plt.legend()
# plt.xlabel("num epochs")
# plt.ylabel("MSE Loss Pruning")
# print(loss_at_epoch[epochs-1])
# https://towardsdatascience.com/training-a-neural-network-using-pytorch-72ab708da210
#https://stackoverflow.com/questions/57949625/with-pytorch-dataloader-how-to-take-in-two-ndarray-data-label

### CHANGE THE LEARNING RATE














    
    
    
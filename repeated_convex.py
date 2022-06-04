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

betas = [10**(j-10) for j in range(10)] # YOU CAN ALSO CHANGE THE BETAS



beta_opt = 1e-4


#betas = betas[0]

def drelu(x):
    return x>=0

losses = []
num_hidden_neurons = []


iteration_len = 2

for betai in betas:
    for ite in range(iteration_len):
        n,d = np.shape(X)
        beta_opt = betai
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
        
        # print(v.value)
        # print(w.value)
        
        
        X = np.multiply(D , (X@(v.value - w.value)))
    losses.append(prob.value)
    
plt.semilogy(betas,losses)
plt.savefig('losses.jpg')



plt.figure()
plt.plot(betas, losses)
plt.semilogx()
plt.semilogy()
plt.xlabel('Beta value')
plt.ylabel('Loss value')
plt.savefig('numhidden.jpg')

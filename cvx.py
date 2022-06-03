# The point of this is to implement an equivalent yet convex representation of the same neural net. 

# Initial tests will be on this super cute data set
import numpy as np
import cvxpy as cp
from math import isclose
import matplotlib.pyplot as plt

np.random.seed(10)
n = 10
d = 3
X = np.random.randn(n,d-1)
X = np.append(X,np.ones((n,1)), axis=1)
y=((np.linalg.norm(X[:,0:d-1],axis=1)>1)-0.5)*2
print(y.shape)
# initialize random NN, put 5 neurons, plant NN
# feed data to NN and take ouput as NN

# n=5
# d=1
# X = np.array([-2, -1, 0, 1, 2]).reshape(n,d)
# y = np.array([1, -1, 1, 1, -1]).reshape(n,d)

beta = 1e-4
betas = [10**(j-10) for j in range(10)]

def drelu(x):
    return x>=0

losses = []
num_hidden_neurons = []

for beta in betas:
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
    cost=cp.sum(cp.pos(1-cp.multiply(y,obj_a)))/n + beta*(cp.mixed_norm(v.T,2,1)+cp.mixed_norm(w.T,2,1))


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

    num_hidden_neurons.append(np.nonzero(alpha)[0].shape[0])
    if beta == 1e-4:
        print(np.nonzero(alpha)[0].shape[0])

plt.semilogy(betas,losses)
plt.savefig('losses.jpg')

plt.figure()
plt.semilogx(betas, num_hidden_neurons)
plt.xlabel('Beta value')
plt.ylabel('Number of hidden neurons')
plt.savefig('numhidden.jpg')



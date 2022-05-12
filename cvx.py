# The point of this is to implement an equivalent yet convex representation of the same neural net. 

# Initial tests will be on this super cute data set
import numpy as np
import cvxpy as cp
from math import isclose

n = 10
d = 3
X = np.random.randn(n,d-1)
X = np.append(X,np.ones((n,1)), axis=1)
y=((np.linalg.norm(X[:,0:d-1],axis=1)>1)-0.5)*2

beta = 1e-4

def relu(x):
    return np.maximum(0,x)
def drelu(x):
    return x>=0

D=np.empty((n,0))

## Finite approximation of all possible sign patterns
for i in range(int(1e2)):
    u=np.random.randn(d,1)
    D=np.append(D,drelu(np.dot(X,u)),axis=1)

D=(np.unique(D,axis=1))
m = D.shape[1]

v = cp.Variable((d, m)) # 3 by 36
w = cp.Variable((d, m))

obj_a = 0
obj_b = 0

constraints = []


obj_a = cp.sum(cp.multiply(D , (X@(v - w))), axis=1)
cost=cp.sum(cp.pos(1-cp.multiply(y,obj_a)))/n+beta*(cp.mixed_norm(v.T,2,1)+cp.mixed_norm(w.T,2,1))


constraints += [cp.multiply(2*D - np.ones((n,m)), (X@v)) >= 0]
constraints += [cp.multiply(2*D - np.ones((n,m)), (X@w)) >= 0]

obj_val = cp.Minimize(cost)

prob = cp.Problem(obj_val, constraints)
prob.solve()

print(prob.value)

# uj is 1x3

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
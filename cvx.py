# The point of this is to implement an equivalent yet convex representation of the same neural net. 

# Initial tests will be on this super cute data set
import numpy as np
import cvxpy as cp

n = 5
d = 1
# X = np.array([-2, -1, 0, 1, 2])
y = np.array([1, -1, 1, 1, -1])
beta = 0

X = np.random.rand(n,d)
mean = np.zeros(d)
cov = np.identity(d)
iteration_len = 10
u = np.random.multivariate_normal(mean,cov,iteration_len)
D = []
for i in range(iteration_len):
    D.append(np.diag(np.int64(X @ u[i,:] > 0)))

v = cp.Variable((iteration_len, d))
w = cp.Variable((iteration_len, d))

obj_a = 0
obj_b = 0

constraints = []

for i in range(iteration_len):
    obj_a += D[i]@X@(v[i,:]- w[i,:]) - y
    obj_b += cp.norm(v[i,:]) + cp.norm(w[i,:])
    constraints += [(2*D[i] - np.eye(n))@X@v[i,:] >= 0]
    constraints += [(2*D[i] - np.eye(n))@X@w[i,:] >= 0]

obj_a = 0.5*cp.norm(obj_a)**2
obj_b = beta*obj_b
obj_val = cp.Minimize(obj_a + obj_b)

prob = cp.Problem(obj_val, constraints)
prob.solve()

print(prob.value/iteration_len)







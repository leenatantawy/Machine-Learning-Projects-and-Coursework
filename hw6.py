import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import math

#1b
num_iterations = 50
A = np.array([[0,0,1,0],[1/2,0,0,0],[1/2,1,0,1], [0,0,0,0]])
b = np.random.rand(A.shape[1])

for i in range(num_iterations):
    b1 = np.dot(A, b)

    bnorm = la.norm(b1,2)

    b = b1 / bnorm

print(b/np.sum(b))


#1c
damping_factor = 0.8
A = np.array([[0,0,1,0],[1/2,0,0,0],[1/2,1,0,1], [0,0,0,0]])
n = A.shape[1]
Google = (damping_factor*A) + (1 - damping_factor) / n*np.ones((n,n))
Google = Google/Google.sum(axis=0)
b = np.random.rand(Google.shape[1])

for i in range(10000):
    b1 = np.dot(Google, b)

    bnorm = la.norm(b1)

    b = b1 / bnorm

print(b/np.sum(b))

#problem 2
n = 200
p = 2
X = 2*(np.random.rand (n , p)- .5)
y = np.sign ( X [: ,1] -( X [: ,0]**2/2+ np.sin ( X [: ,0]*7) /2) )
plt . figure (1)
plt . scatter ( X [: , 0] , X [: , 1] , 50 , c = y )
plt . colorbar ()
plt . xlabel ('feature 1')
plt . ylabel ('feature 2')
plt . title ('2d training samples colored by label')
plt . show ()

sigma = .05
lam = 1
norms2 = (np.array (la.norm (X , axis =1)).T ) **2 # squared norm ofeach training sample
innerProds = X@X.T
dist2 = np.matrix(norms2).T@np.ones([1 ,n]) + np.ones ([n ,1])@np.matrix(norms2) -2* innerProds # squared distances between eachpair of training samples
K = np.zeros([200,200])
for i in range(n):
    for j in range(n):
        K[i,j] = math.exp(-(dist2[i,j])/(2*sigma))

ones = np.identity(200)
alpha = la.inv(K + lam * ones)@y
yhat = K @ alpha

y2 = np.array(np.sign(yhat))
plt . figure (2)
plt . scatter ( X[: , 0] , X[: , 1] , 50 , c = y2 )
plt . colorbar ()
plt . xlabel ('feature 1')
plt . ylabel ('feature 2')
plt . title ('2d training samples colored by PREDICTED label')
plt . show ()
ntest = 2000
Xtest = 2*(np.random.rand(ntest, p) -.5)
norms2_test = (np.array(la.norm (Xtest , axis =1)).T) **2
innerProds_test = Xtest@X.T
dist2_test = np.matrix(norms2_test).T@np.ones([1 , n]) + np.ones ([ntest ,1])@np.matrix (norms2) -2* innerProds_test
K_test = np.zeros([ntest,n])
for i in range(ntest):
    for j in range(n):
        K_test[i,j] = math.exp(-(dist2_test[i,j])/(2*sigma))

ytest = K_test @ alpha

plt . figure (3)
plt . scatter (Xtest [: ,0] , Xtest [: ,1] , 50 , c = np.array(ytest) )
plt . colorbar ()
plt . xlabel ('feature 1')
plt . ylabel ('feature 2')
plt . title ('2d test samples colored by PREDICTED label ( before taking sign )')
plt . show ()
# python
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la


# load the data matrix X
d_jest = sio.loadmat('jesterdata (1).mat')
X = d_jest['X']
# load known ratings y and true ratings truey
d_new = sio.loadmat('newuser.mat')
y = d_new['y']
true_y = d_new['truey']
# total number of joke ratings should be m = 100 , n = 7200
m , n = X.shape
# train on ratings we know for the new user
train_indices = np.squeeze(y != -99)
num_train = np.count_nonzero(train_indices)
# test on ratings we don ’t know
test_indices = np.logical_not(train_indices)
num_test = m - num_train
X_data = X[train_indices, 0:20]
y_data = y[train_indices]
y_test = true_y[test_indices]



# 2a


# solve for weights
U, S, VT = la.svd(X_data, full_matrices = True)

print(S)

S_diag = np.diag(S)
print(S_diag)
S0 = np.zeros((20,1))
Sinv = (la.inv(S_diag.T @ S_diag)) @ S_diag.T

for i in range(5):
    Sadd = np.hstack((Sinv,S0))
    Sinv = Sadd
    i+=1
    
print(Sinv.shape)

print(Sinv)


weights = VT.T @ Sinv @ U.T @ (y_data)

print(weights)
# compute predictions
y_hat_train = X_data@weights

# measure performance on training jokes


#For first 25 indices
avg_error = y_hat_train - y_data

avgerr_train = la.norm(avg_error)
print(avgerr_train)





# display results
ax1 = plt . subplot (121)
sorted_indices = np . argsort ( np . squeeze ( y_data ) )
ax1 . plot (
    range ( num_train ) , y_data [ sorted_indices ] , 'b.' ,
    range ( num_train ) , y_hat_train [ sorted_indices ] , 'r.'
    )
ax1 . set_title ( 'prediction of known ratings ( trained with 20 users )')
ax1 . set_xlabel ( 'jokes ( sorted by true rating )')
ax1 . set_ylabel ('rating')
ax1 . legend ([ 'true rating' , 'predicted rating' ] , loc = 'upper left')
ax1 . axis ([0 , num_train , -15 , 10])
print ( " Average l_2 error ( train ) : " , avgerr_train )
plt.show()
# measure performance on unrated jokes

X_test = X[test_indices, 0:20]

y_hat_test = X_test@weights
print(y_hat_train.shape)
print(y_hat_test.shape)
print(y_test.shape)

avg_error = y_hat_test - y_test

avgerr_test = la.norm(avg_error)
print(avgerr_test)

# display results
ax2 = plt . subplot (122)
sorted_indices = np . argsort ( np . squeeze ( y_test ) )
ax2 . plot (
    range ( num_test ) , y_test [ sorted_indices ] , 'b.' ,
    range ( num_test ) , y_hat_test [ sorted_indices ] , 'r.'
)
ax2 . set_title (' prediction of unknown ratings ( trained with 20 users )')
ax2 . set_xlabel ( 'jokes ( sorted by true rating ) ')
ax2 . set_ylabel ( 'rating' )
ax2 . legend ([ 'true rating' , 'predicted rating'] , loc = 'upper left')
ax2 . axis ([0 , num_test , -15 , 10])
print ( " Average l_2 ( test ) : " , avgerr_test )
plt.show ()


#2b

U, S, VT = la.svd(X, full_matrices = True)
print(U.shape)
print(S.shape)
print(VT.shape)


S_diag = np.diag(S)
print(S_diag.shape)
S0 = np.zeros((20,1))
Sinv = (la.inv(S_diag.T @ S_diag)) @ S_diag.T


Sinv = np.zeros((7200,100))

for i in range(len(S)):
    Sinv[i][i] = 1/(S[i])


print(Sinv.shape)

print(Sinv)

#find the weights
weights = VT.T @ Sinv @ U.T @ (y)
print(VT.T.shape)
print(Sinv.shape)
print(U.T.shape)


#find prediction
y_pred = X@weights

#find avg_error with norm of predictions
avg_error = y_pred - y
avgerr_all = la.norm(avg_error)
print(avgerr_all)


#2c

#Use weight of finding full matrix weights , use 72000 x 2 matrix, against weights of full matrix and then get those predicitions and then one person would be the average of those predictions


#2d plot i vs sigma i to plot the svd values

U, S, V = la.svd(X, full_matrices= False)

x = []
y = []

for i in range(len(S)):
    x.append(i)
    y.append(S[i])

plt.plot(x,y, 'ro')
plt.show()


#2e highest 3 sigma value directions, use pca
# imagine a row is a vector in p space, then take that vector from Rp to R3 so go from X in R nxp to something in R n x 3 using the 3 best svd (multiply by pX3) 
# first 3 columns of u or v (multply by columns of v and not by Vt), and then v is the projection matrix

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

#column vectors
U,S,VT = la.svd(X, full_matrices = False)

U_vis = U[:,0:3]
V_vis = VT.T[:,0:3]
x_vis_u = U_vis.T@X
x_vis_v = X@V_vis

fig1 = plt.figure()
axes = axes3d(fig1)
axes3d.scatter(x = x_vis_u[0,:], y = x_vis_u[1,:], z = x_vis_u[2,:], s=20, c=None, depthshade=True)
plt.show()

#row vectors
U,S,VT = la.svd(X, full_matrices = False)

U_vis = U[:,0:3]
V_vis = VT[:,0:3]
x_vis_u = U_vis.T@X
x_vis_v = X@V_vis

fig1 = plt.figure()
axes3d(fig1)
axes3d.scatter(x = x_vis_v[0,:], y = x_vis_v[1,:], z = x_vis_v[2,:], s=20, c=None, depthshade=True)
plt.show()
#2f

import numpy as np

num_iterations = 50
B = X@X.T
b = np.random.rand(B.shape[1])


for i in range(num_iterations):
    b1 = np.dot(B, b)

    bnorm = la.norm(b1,2)

    b = b1 / bnorm

print("left singular:", b)



num_iterations = 50
A = X.T@X
b = np.random.rand(B.shape[1])

for i in range(num_iterations):
    b1 = np.dot(A, b)

    bnorm = la.norm(b1,2)

    b = b1 / bnorm

print("right singular:", b)


U, S, VT = la.svd(X, full_matrices = True)
print(U[:,0])
print(VT.T[:,0])



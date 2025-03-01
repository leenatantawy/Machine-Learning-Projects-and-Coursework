import scipy . io as sio
import numpy.linalg as la
import numpy as np
# #### Part a #####
# load the training data X and the training labels y
matlab_data_file = sio . loadmat ('face_emotion_data.mat')
X = matlab_data_file ['X']
y = matlab_data_file ['y']
# n = number of data points
# p = number of features
n , p = np . shape ( X )
# Solve the least - squares solution . w is the list of
# weight coefficients
w = la.inv(X.T@X)@X.T@y
print ( w )

yhat_real = X@w
yhat_sign = np.sign(yhat_real)
errors = yhat_sign != y
print(sum(abs(errors))/16)
    
# divide X into 8 equal subsets
size = n // 8
subsets = []
print(X.shape)
for i in range(8):
    start = i * size
    end = start + size
    subset = X[start:end, :]
    subsets.append(subset)

size = n // 8
ysubsets = []
for i in range(8):
    start = i * size
    end = start + size
    ysubset = y[start:end]
    ysubsets.append(ysubset)

error_rates = []
for i in range(8):
    test = []
    ytest = []
    for j in range(8):
        if not i == j:
            test.append(subsets[j])
            ytest.append(ysubsets[j])
    X = np.concatenate(test)
    y = np.concatenate(ytest)
    
    print(X.shape)
    print(y.shape)
    w = la.inv(X.T@X)@X.T@y
    ypred = np.dot(X,w)

    yhat_real = X@w
    yhat_sign = np.sign(yhat_real)
    errors = yhat_sign != y
    error_rate = (sum(abs(errors))/16)
    error_rates.append(error_rate)
        
    print(w)

print(np.mean(error_rates))
    
    
       


#PROBLEM 5

import numpy as np
import scipy . io as sio
import matplotlib.pyplot as plt
# load x and y vectors
data = sio.loadmat("polydata.mat")
x = data ["x"]
y = data ["y"]
# n = number of data points
# N = number of points to use for interpolation
# z_test = points where interpolant is evaluated
# p = array to store the values of the interpolated polynomials
n = x.size
N = 100
z_test = np.linspace(np.min(x), np.max (x), N )
p = np.zeros ((3, N))
for d in [1 , 2 , 3]:
    X = np.zeros((n, d+1))
    for i in range(n):
        for k in range(0, d+1):
            X[i][k] = x[i]**k
    w = la.inv(X.T@X)@X.T@y
    p[d-1] = np.polyval(w[::-1], z_test)

# generate X- matrix for this choice of d
# solve least - squares problem . w is the list
# of polynomial coefficients
# evaluate best -fit polynomial at all points z_test ,
# and store the result in p
# NOTE ( optional ): this can be done in one line
# with the polyval command !
# plot the datapoints and the best -fit polynomials
plt.plot (x , y , ".", z_test , p [0 , :] , z_test , p [1 , :] , z_test , p
[2 , :] , linewidth =2)
plt.legend (["data", "d=1", "d=2", "d=3"] , loc = "upper left")
plt.title ("best fit polynomials of degree 1 , 2 , 3")
plt.xlabel ("x")
plt.ylabel ("y")
plt.show ()

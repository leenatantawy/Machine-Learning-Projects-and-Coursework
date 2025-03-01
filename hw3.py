import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.io


# #testing 1c 
# X = np.array([[1,0], [0,3/5], [0,4/5]])
# y = np.array([[1], [6], [2]])

# np.shape(X)
# np.shape(y)
# inv = la.inv(X.T@X)

# res = X@inv@X.T@y
# print(res)


#QUESTION 3
# load data, make sure ‘fisheriris.mat‘ is in your working directory
data = scipy.io.loadmat("fisheriris.mat")
# training data
# training data
X = data['meas']
y = data['species']
np.shape(X)
np.shape(y)

y_num = np.zeros(150)
for i in range(150):
    if y[i] == 'setosa':
        y_num[i] = 0
    if y[i] == 'versicolor':
        y_num[i] = 1
    if y[i] == 'virginica':
        y_num[i] = 2

# number of random trials
N = 10000
# array to store errors
errs = np.zeros(N)
# size of training set
num_train = 40
error_rates = []

for i in np.arange(1, N+1):
# for each experiment, randomly pick training and holdout sets
    idx_train = np.zeros(0, dtype=np.intp)
    idx_holdout = np.zeros(0, dtype=np.intp)
    for label_type in range(3):
        r = np.random.permutation(50)
        idx_train = np.concatenate((idx_train, 50 * label_type + r[:num_train]))
        idx_holdout = np.concatenate((idx_holdout, 50 * label_type + r[num_train:]))
# divide data and labels into subsets
    Xt = X[idx_train, :]
    yt = y_num[idx_train]
    Xh = X[idx_holdout]
    yh = y_num[idx_holdout]

    w = la.inv(Xt.T@Xt)@Xt.T@yt
    y_real = Xh@w
    mistakes = 0
    for i in range(len(y_real)):
        if y_real[i] < 0.5:
            y_real[i] = 0
        elif y_real[i] >= 0.5 and y_real[i] < 1.5:
            y_real[i] = 1
        elif y_real[i] >= 1.5:
            y_real[i] = 2
        if y_real[i] != yh[i]:
            mistakes+=1
    error_rate = mistakes/30
    error_rates.append(error_rate)
print(np.mean(error_rates))


#3c
import numpy as np
import scipy.io
# load data, make sure ‘fisheriris.mat‘ is in your working directory
data = scipy.io.loadmat("fisheriris.mat")
# training data
X = data["meas"]
y = data["species"]
# YOUR CODE BELOW (process and assign numerical values to ‘y‘ according
y = ...
# number of random trials
N = 10000
# size of training set
max_num_train = 40
# array to store errors
errs = []
for i in range(40):
    num_train = i + 1
    for j in range(N):
# for each experiment, randomly pick training and holdout sets
        idx_train = np.zeros(0, dtype=np.intp)
        idx_holdout = np.zeros(0, dtype=np.intp)
        for label_type in range(3):
            r = np.random.permutation(50)

            idx_train = np.concatenate((idx_train, 50 * label_type + r[:num_train]))
            idx_holdout = np.concatenate((idx_holdout, 50 * label_type + r[num_train:]))
        # divide data and labels into subsets
        Xt = X[idx_train, :]
        yt = y_num[idx_train]
        Xh = X[idx_holdout]
        yh = y_num[idx_holdout]
        ##CHECK BELOW, gettung same answer as before
        w = la.inv(Xt.T@Xt)@Xt.T@yt
        y_real = Xh@w
        mistakes = 0
        for i in range(len(y_real)):
            if y_real[i] < 0.5:
                y_real[i] = 0
            elif y_real[i] >= 0.5 and y_real[i] < 1.5:
                y_real[i] = 1
            elif y_real[i] >= 1.5:
                y_real[i] = 2
            if y_real[i] != yh[i]:
                mistakes+=1
    avg_error = mistakes/(150 - (3*num_train))
    avg_errs = []
    avg_errs.append(avg_error)
    errs.append(avg_errs)


errors = []
for i in range(40):
    error = np.mean(errs[i])
    errors.append(error)

x = []
i = 3
while i< 123:
    x.append(i)
    i += 3
print(x)
    
plt.scatter(x, y= errors)
plt.show()

#3d

# load data, make sure ‘fisheriris.mat‘ is in your working directory
data = scipy.io.loadmat("fisheriris.mat")
# training data
# training data
X = data['meas']
y = data['species']
np.shape(X)
np.shape(y)

X_d = np.zeros((150,3))
X_d[:,0] = X[:,0]
X_d[:,1] = X[:,1]
X_d[:,2] = X[:,2]
print(X_d)

y_num = np.zeros(150)
for i in range(150):
    if y[i] == 'setosa':
        y_num[i] = 0
    if y[i] == 'versicolor':
        y_num[i] = 1
    if y[i] == 'virginica':
        y_num[i] = 2

# number of random trials
N = 10000
# array to store errors
errs = np.zeros(N)
# size of training set
num_train = 40
error_rates = []

for i in np.arange(1, N+1):
# for each experiment, randomly pick training and holdout sets
    idx_train = np.zeros(0, dtype=np.intp)
    idx_holdout = np.zeros(0, dtype=np.intp)
    for label_type in range(3):
        r = np.random.permutation(50)
        idx_train = np.concatenate((idx_train, 50 * label_type + r[:num_train]))
        idx_holdout = np.concatenate((idx_holdout, 50 * label_type + r[num_train:]))
# divide data and labels into subsets
    Xt = X_d[idx_train, :]
    yt = y_num[idx_train]
    Xh = X_d[idx_holdout]
    yh = y_num[idx_holdout]

    w = la.inv(Xt.T@Xt)@Xt.T@yt
    y_real = Xh@w
    mistakes = 0
    for i in range(len(y_real)):
        if y_real[i] < 0.5:
            y_real[i] = 0
        elif y_real[i] >= 0.5 and y_real[i] < 1.5:
            y_real[i] = 1
        elif y_real[i] >= 1.5:
            y_real[i] = 2
        if y_real[i] != yh[i]:
            mistakes+=1
    error_rate = mistakes/30
    error_rates.append(error_rate)

print(np.mean(error_rates))


#3e
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d') # 3d plotting
# YOUR CODE BELOW (generate a 3d scatter plot, you may find the ax)
ax.scatter(xs = X_d[:,0], ys = X_d[:,1], zs = X_d[:,2])
# rotating (change the following parameters for different angles)
elevation = 10
azimuth = 10
# rotates 3d scatter plot
ax.view_init(elev=elevation, azim=azimuth)
plt.show()




#3f




#Question 4
import numpy as np
import numpy.matlib as mat
import numpy.linalg as la
import matplotlib.pyplot as plt
# function f
def f(w):
    return w*w*np.cos(w)-w
def grad_f(w):
    return 2*w*np.cos(w)-w*w*np.sin(w)-1
n = 100
p = 1
w = np.linspace(-12,12,num=n)
myf = f(w)
plt.plot(w, myf, linewidth =2)
plt.xlabel("w")
plt.ylabel("f(w)")
plt.show()


tau = .01
max_iter = 10
w_hat = np.matrix(np.zeros((max_iter+1,1)))
f_hat = np.matrix(np.zeros((max_iter+1,1)))
w_hat[0] = 7 # initial value of w
f_hat[0] = f(w_hat[0]) # corresponding initial value of f
for k in range(max_iter):
    w_hat[k+1] = w_hat[k] - tau*grad_f(w_hat[k])
    f_hat[k+1] = f(w_hat[k+1])
    # implement gradient descent here
    # store new w in w_hat[k+1]
    # store new f(w) in f_hat[k+1]
plt.plot(w,myf,linewidth=2)
plt.plot (w_hat ,f_hat,"-*",linewidth =2)
plt.xlabel ("w")
plt.ylabel ("f(w)")
plt.show ()
print(w_hat)

#increase step size test
tau = .15
max_iter = 10
w_hat = np.matrix(np.zeros((max_iter+1,1)))
f_hat = np.matrix(np.zeros((max_iter+1,1)))
w_hat[0] = 7 # initial value of w
f_hat[0] = f(w_hat[0]) # corresponding initial value of f
for k in range(max_iter):
    w_hat[k+1] = w_hat[k] - tau*grad_f(w_hat[k])
    f_hat[k+1] = f(w_hat[k+1])
    # implement gradient descent here
    # store new w in w_hat[k+1]
    # store new f(w) in f_hat[k+1]
plt.plot(w,myf,linewidth=2)
plt.plot (w_hat ,f_hat,"-*",linewidth =2)
plt.xlabel ("w")
plt.ylabel ("f(w)")
plt.show ()
print(w_hat)


#decrease step size test
tau = .001
max_iter = 10
w_hat = np.matrix(np.zeros((max_iter+1,1)))
f_hat = np.matrix(np.zeros((max_iter+1,1)))
w_hat[0] = 7 # initial value of w
f_hat[0] = f(w_hat[0]) # corresponding initial value of f
for k in range(max_iter):
    w_hat[k+1] = w_hat[k] - tau*grad_f(w_hat[k])
    f_hat[k+1] = f(w_hat[k+1])
    # implement gradient descent here
    # store new w in w_hat[k+1]
    # store new f(w) in f_hat[k+1]
plt.plot(w,myf,linewidth=2)
plt.plot (w_hat ,f_hat,"-*",linewidth =2)
plt.xlabel ("w")
plt.ylabel ("f(w)")
plt.show ()
print(w_hat)

#changing inital value test
tau = .001
max_iter = 10
w_hat = np.matrix(np.zeros((max_iter+1,1)))
f_hat = np.matrix(np.zeros((max_iter+1,1)))
w_hat[0] = -4 # initial value of w
f_hat[0] = f(w_hat[0]) # corresponding initial value of f
for k in range(max_iter):
    w_hat[k+1] = w_hat[k] - tau*grad_f(w_hat[k])
    f_hat[k+1] = f(w_hat[k+1])
    # implement gradient descent here
    # store new w in w_hat[k+1]
    # store new f(w) in f_hat[k+1]
plt.plot(w,myf,linewidth=2)
plt.plot (w_hat ,f_hat,"-*",linewidth =2)
plt.xlabel ("w")
plt.ylabel ("f(w)")
plt.show ()
print(w_hat)


#4b

import numpy as np
import numpy.matlib as mat
import numpy.linalg as la
import matplotlib.pyplot as plt
n = 200
x = np.linspace(-1,1,n)
y = np.matrix(np.cos(3*x)).T
X = np.matrix([x**0, x**1, x**2, x**3, x**4, x**5]).T
alt_w_hat = la.inv(X.T@X)@X.T@y
tau = 2.8e-3
max_iter = 5000
w_hat = np.matrix(np.zeros((6,max_iter+1)))
w_hat[:,0] = np.zeros([6,1])
plt.plot(x,y,linewidth =2,label='y = cos(x)')
plt.plot(x,X@alt_w_hat,label='Least squares fit')
plt . xlabel ('x')
plt . ylabel ('y')
for k in range(max_iter):
    w_hat[:,k+1] = w_hat[:,k] - 2*tau*X.T@(X@w_hat[:,k]-y)
    
ktype = np.logspace(0,np.log10(max_iter/2),5,base=10).astype(int)
for k in ktype:
    plt.plot(x,X@w_hat[:,k],'--',linewidth=1,label='Grad descent est, k = '+ format(k,'02d'))
    plt.plot(x,X@w_hat[:,max_iter],'-',linewidth=2,label='Final estimate'+ format(max_iter,'02d')+ ' iterations')
plt.legend()
plt.show()
print(w_hat)

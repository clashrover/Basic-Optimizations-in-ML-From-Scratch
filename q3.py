import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 

import math


def plot_linear(y,x,theta):
    m,n = np.shape(x)
    # print(x)
    for i in range(m):
        if y[i][0]==0:
            plt.plot(x[i,1],x[i,2],'bv',markersize=5)
        else:
            plt.plot(x[i,1],x[i,2],'ro',markersize=5)
    
    lpx = np.linspace(-2,2,100)
    lpy =  (theta[1][0]*lpx)
    lpy = lpy + theta[0][0]
    lpy = lpy/ theta[2][0]
    lpy = lpy*-1 

    plt.plot(lpx,lpy,'-g',label = 'Logistic Regression')
    plt.title('Graph of Logistic Regression')
    plt.xlabel('X1', color='#1C2833')
    plt.ylabel('X2', color='#1C2833')
    plt.show()

def hypothesis(x,theta):
    n,m = np.shape(theta)
    x1 = np.zeros((m,n))
    x1[0:m][0:n] = x
    x=x1
    tx = np.matmul(x,theta)
    # print(tx)
    tx = tx*-1
    # print(tx[0])
    e = math.exp(tx[0][0]) +1 
    e =1/e
    return e

def gradient(y,x,theta):
    m,n = np.shape(x)
    g = np.zeros((n,1))
    # print(x)
    # print(x[0])
    for i in range(m):
        x1 = np.zeros((1,n))
        x1[0:1][0:n] = x[i]
        x1 = np.transpose(x1)
        # print(x1)
        g = g + ((hypothesis(x[i],theta)-y[i][0])*x1)

    g=g*-1
    # print(g)
    return g

def hessian(y,x, theta):
    m,n = np.shape(x)
    g = np.zeros((n,n))
    
    for i in range(m):
        p = hypothesis(x[i],theta)
        
        x1 = np.zeros((1,n))
        x1[0:1][0:n] = x[i]
        x1 = np.transpose(x1)

        xxt = np.matmul(x1,np.transpose(x1))
        # print(xxt)
        g = g + (p*(1-p)* xxt)
        # print(g)
    # print(g)
    return g

# convergence test
def converge(e1, e,lim):
    return abs(e1-e) < lim

def likelihood(y,x,theta):
    m,n = np.shape(x)
    l=0
    for i in range(m):
        p = hypothesis(x[i],theta)
        l += + y[i][0]*math.log(p)+ (1-y[i][0])*math.log(1-p)

    return l

def logistic_reg(y,x,k):
    # optimize LL(theta) using newton's method
    m,n = np.shape(x)
    theta = [[0],[0],[0]]
    
    ll = likelihood(y,x,theta)
    lim = 0.000001
    t=0
    while True:
        g = gradient(y,x,theta)
        h = hessian(y,x,theta)
        # print(g)
        # print(h)
        theta1 = theta + (k* np.matmul(np.linalg.inv(h),g))
        t+=1
        # print(theta1)
        ll1 = likelihood(y,x,theta1)  
        if converge(ll1,ll,lim):
            ll = ll1
            theta = theta1
            break  
        theta = theta1
        ll = ll1
    
    print(theta)
    print(ll)
    print(t)
    plot_linear(y,x,theta)

def normalise(x):
    n,m = np.shape(x)
    # print(x)
    x[0][0:m] = x[0][0:m]- np.mean(x[0][0:m])
    x[0][0:m] = x[0][0:m]/ np.std(x[0][0:m])
    x[1][0:m] = x[1][0:m]- np.mean(x[1][0:m])
    x[1][0:m] = x[1][0:m]/ np.std(x[1][0:m])
    # print(x)
    
def main():
    with open('logisticX.csv') as fx,open('logisticY.csv') as fy:
        lines = (line for line in fx if not line.startswith('#'))
        x = np.loadtxt(lines, delimiter=',')
        # f1 = np.transpose(f1)
        lines = (line for line in fy if not line.startswith('#'))
        y = np.loadtxt(lines, delimiter=',')
        # f2 = np.transpose(f2)
        
        x = np.transpose(x)
        normalise(x)
        n,m = np.shape(x)

        x1 = np.ones((1,m))
        x = np.vstack((x1,x))
        y1 = np.zeros((1,m))
        y1[0][0:m]=y
        y1=np.transpose(y1)
        y=y1
        # print(y)
        # print(x)
        x = np.transpose(x)
        logistic_reg(y,x,0.5)
    


main()
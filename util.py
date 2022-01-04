

import numpy as np
import sys
import os



# ---------------------------------------------------------------------
# Auxillary Functions

# calculate gradient of JΘ
# direct formula
def gradient(y,x,theta):
    m,n = np.shape(x)
    r = np.matmul(np.transpose(x),x)
    r = np.matmul(r,theta)
    d = np.matmul(np.transpose(x),y)
    g = r-d
    g = g/m
    return g

# convergence test
def converge(e1, e,lim):
    return abs(e1-e) < lim

#erro direc formula
def error(y,x,theta):
    m,n = np.shape(x)
    r = np.matmul(x,theta)- y
    e = np.matmul(np.transpose(r),r)
    e = e/(2*m)
    return e[0][0]

# y and x are grid, we need to retur a grid that has error value for corresponding y,x point
def grad(t1,t0,y,x):
    z = np.zeros(np.shape(t1))
    r,c = np.shape(t1)
    for i in range(r):
        for j in range(c):
            th = np.zeros((2,1))
            th[0][0]=t1[i][j]
            th[1][0]=t0[i][j]
            z[i][j] = error(y,x,th)
    return z
#-------------------------------------------------------------------------------------
# linear regression using gradient descent

def linear_regression(y,x,k):
    #initialisation
    n,m = np.shape(x)
    theta = np.zeros((n,1))
    t=0 
    x=np.transpose(x)           
    e= error(y,x,theta)
    t=0
    theta0_list = [theta[0][0]]
    theta1_list = [theta[1][0]]
    error_list =[e]
    lim = 0.00000001
    #loop
    while True:
        # print(e)
        p = gradient(y,x,theta)
        theta1 = theta - (k*p)                  #the formula
        theta0_list.append(theta[0][0])         #store theta for plottting later
        theta1_list.append(theta[1][0])
        
        e1 = error(y,x,theta1)                  #find error
        error_list.append(e1)

        t+=1
        if converge(e1, e,lim):                 #check for convergence
            theta = theta1
            break
        theta = theta1
        e = e1

    
    return lim, theta, theta0_list, theta1_list, error_list
    
# -----------------------------------------------------------------------------
# Sampling and Stochastic Gradient Descent

import math

def generate():
    theta = np.array([[3],[1],[2]])
    
    sample_size = 1000000
    x = np.zeros((sample_size,3))
    
    x[0:sample_size,0] = np.ones(sample_size)
    mu=3.0
    sig = 4.0
    x[0:sample_size,1] = np.random.normal(mu, math.sqrt(sig),sample_size)
    mu=-1.0
    sig=4.0
    x[0:sample_size,2] = np.random.normal(mu, math.sqrt(sig),sample_size)
    x = np.transpose(x)
    
    y = np.zeros((sample_size,1))
    
    y[0:sample_size,0] = np.matmul(np.transpose(theta),x)
    
    mu =0.0
    sig = 2
    normal = np.random.normal(mu,math.sqrt(sig),sample_size)
    
    y=np.transpose(y)
    y[0,0:sample_size] = y[0,0:sample_size] + normal
    
    y = np.transpose(y)
    return (x,y)


def shuffle(y,x):
    """
    To shuffle the data set
    """
    shuf  = np.vstack((np.transpose(y),x))
    shuf = np.transpose(shuf)
    np.random.shuffle(shuf)
    shuf = np.transpose(shuf)
    y,x = np.split(shuf,[1])
    y=np.transpose(y)
    return y,x



def sgd(y,x,r,k,avg,l):
    # x, y = generate()
    n,m = np.shape(x)
    # form the x matrix with each row as xi that is a matrix of size 2*1
    theta = np.zeros((n,1))
    
    t=0 #time                 
    

    epoch=0

    e_old = 0
    e =0

    epoch_limit = 10000
    y,x = shuffle(y,x)
    x=np.transpose(x)    #design matrix

    theta_list = []   #for plotting

    while True:
        bln=False
        for b in range(math.floor(m/r)):
            p = gradient(y[b*r:r*(b+1)],x[b*r:r*(b+1)],theta)
            theta1 = theta - (k*p)
            e+= error(y[b*r:r*(b+1)],x[b*r:r*(b+1)],theta1)
            t+=1
            theta = theta1
            theta_list.append(theta) 
            
            if t%avg == 0:
                e=e/avg
                # print(e,e_old)
                if converge(e,e_old,l):
                    # print("-----")
                    bln=True
                    break
                e_old = e
                # print(e)
                e=0
        
        if epoch > epoch_limit:
            break
        if bln:
            break
        epoch+=1
        
    theta_list.append(theta)
    return theta,theta_list
    
    # -------------------------------------------------------------------------


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


def gradient_logistic(y,x,theta):
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

def likelihood(y,x,theta):
    m,n = np.shape(x)
    l=0
    for i in range(m):
        p = hypothesis(x[i],theta)
        l += + y[i][0]*math.log(p)+ (1-y[i][0])*math.log(1-p)

    return l

def logistic_reg(y,x,k,lim):
    # optimize LL(theta) using newton's method
    m,n = np.shape(x)
    theta = [[0],[0],[0]]
    
    ll = likelihood(y,x,theta)
    
    t=0
    while True:
        g = gradient_logistic(y,x,theta)
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
    
    return theta
    # print(theta)
    # print(ll)
    # print(t)
    # plot_linear(y,x,theta)

def normalise(x):
    n,m = np.shape(x)
    # print(x)
    for i in range(n):
        x[i][0:m] = x[i][0:m]- np.mean(x[i][0:m])
        x[i][0:m] = x[i][0:m]/ np.std(x[i][0:m])
    return x
        
    
# -----------------------------------------------------------------------------------

def gda(y,x):
    n,m = np.shape(x)
    u0=np.zeros((n,1))
    u1=np.zeros((n,1))
    sigma = np.zeros((n,n))
    x=np.transpose(x)
    # print(x[0][0:n])
    t0=0
    t1=0
    for i in range(m):
        if y[i][0] == 0:
            x1 = np.zeros((1,n))
            x1[0:1,0:n]=x[i]
            # print(np.transpose(x1))
            # print(u0)
            u0 = u0 + np.transpose(x1)
            # print(u0)
            t0+=1
        else:
            x1 = np.zeros((1,n))
            x1[0:1,0:n]=x[i]
            u1 = u1 + np.transpose(x1)
            t1+=1
    
    u0 = u0/t0
    u1 = u1/t1
    # print(u0)
    # print(u1)
    # print(x)
    for i in range(m):
        if y[i][0]==0:
            x1 = np.zeros((1,n))
            x1[0:1,0:n]=x[i]
            x1=np.transpose(x1)
            sigma = sigma + np.matmul((x1-u0), np.transpose(x1-u0))
        else:
            x1 = np.zeros((1,n))
            x1[0:1,0:n]=x[i]
            x1=np.transpose(x1)
            sigma = sigma + np.matmul((x1-u1), np.transpose(x1-u1))
    
    sigma = sigma/m
    # print(sigma)
    # print(x)
    phi = t1/m
    # print(phi)
    c1 = math.log(phi/(1-phi))
    c2 = np.matmul(np.transpose(u1),np.matmul(np.linalg.inv(sigma),u1))
    c2 = c2 - np.matmul(np.transpose(u0),np.matmul(np.linalg.inv(sigma),u0))
    c2=c2/2
    c2 = -1*c2[0][0]
    coeff = np.matmul(np.linalg.inv(sigma),(u1-u0))
    const = c1+c2

    return u0, u1 , sigma, const, coeff
    # print(coeff)
    # plot_graph(y,x)
    # plot_linear(y,x,coeff,const)
    # plot_quad

def general_gda(y,x):
    n,m = np.shape(x)
    u0=np.zeros((n,1))
    u1=np.zeros((n,1))

    x=np.transpose(x)
    # print(x[0][0:n])
    t0=0
    t1=0
    for i in range(m):
        if y[i][0] == 0:
            x1 = np.zeros((1,n))
            x1[0:1,0:n]=x[i]
            # print(np.transpose(x1))
            # print(u0)
            u0 = u0 + np.transpose(x1)
            # print(u0)
            t0+=1
        else:
            x1 = np.zeros((1,n))
            x1[0:1,0:n]=x[i]
            u1 = u1 + np.transpose(x1)
            t1+=1
    
    u0 = u0/t0
    u1 = u1/t1
    # print("u0",u0)
    # print("u1",u1)
    # print(x)
    sigma0 = np.zeros((n,n))
    sigma1 = np.zeros((n,n))
    for i in range(m):
        if y[i][0]==0:
            x1 = np.zeros((1,n))
            x1[0:1,0:n]=x[i]
            x1=np.transpose(x1)
            sigma0 = sigma0 + np.matmul((x1-u0), np.transpose(x1-u0))
        else:
            x1 = np.zeros((1,n))
            x1[0:1,0:n]=x[i]
            x1=np.transpose(x1)
            sigma1 = sigma1 + np.matmul((x1-u1), np.transpose(x1-u1))
    
    sigma0 = sigma0/t0
    sigma1 = sigma1/t1
    # print("sigma0",sigma0)
    # print("sigma1",sigma1)
    phi = t1/m
    # print(phi)
    c1 = -1*math.log(phi/(1-phi))
    c2 = np.matmul(np.transpose(u1),np.matmul(np.linalg.inv(sigma1),u1))
    c2 = c2 - np.matmul(np.transpose(u0),np.matmul(np.linalg.inv(sigma0),u0))
    c2 = c2[0][0] /2
    c3 = np.matmul(np.linalg.inv(sigma1),u1) - np.matmul(np.linalg.inv(sigma0),u0)
    c3 = c3 * (-1)
    c4 = np.linalg.inv(sigma1)-np.linalg.inv(sigma0)
    c4 = c4/2
    # plot_quadratic(c4,c3,c2,c1,y,x) 
    return u0,u1,sigma0,sigma1,c4,c3,c2,c1

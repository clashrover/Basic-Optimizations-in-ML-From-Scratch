import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 
import math


def plot_graph(y,x):
    m,n = np.shape(x)
    # print(x)
    for i in range(m):
        if y[i][0]==0:
            plt.plot(x[i,0],x[i,1],'bv',markersize=5)
        else:
            plt.plot(x[i,0],x[i,1],'ro',markersize=5)
    

    plt.title('Graph of GDA')
    plt.xlabel('X1', color='#1C2833')
    plt.ylabel('X2', color='#1C2833')
    plt.show()

def plot_linear(y,x,theta,const):
    m,n = np.shape(x)
    # print(x)
    for i in range(m):
        if y[i][0]==0:
            plt.plot(x[i,0],x[i,1],'bv',markersize=5)
        else:
            plt.plot(x[i,0],x[i,1],'ro',markersize=5)
    
    lpx = np.linspace(-2,2,100)
    lpy =  (theta[0][0]*lpx)
    lpy = lpy + const
    lpy = lpy/ theta[1][0]
    lpy = lpy*-1 

    plt.plot(lpx,lpy,'-g',label = 'GDA')
    plt.title('Graph of GDA')
    plt.xlabel('X1', color='#1C2833')
    plt.ylabel('X2', color='#1C2833')
    plt.show()


def plot_quadratic(c4,c3,c2,c1,y1,x1):
    m,n = np.shape(x1)
    # print(x)


    lpx = np.linspace(-2,2,100)
    lpy = np.linspace(-2,2,100)
    X, Y = np.meshgrid(lpx, lpy)
    F = c4[0][0]*X**2 + (c4[1][0] + c4[0][1])*X*Y + c4[1][1]*Y**2 + c3[0][0]*X + c3[1][0]*Y + c2+c1

    fig,ax = plt.subplots()
    ax.contour(X, Y, F, levels=[0]) # take level set corresponding to 0
    
    for i in range(m):
        if y1[i][0]==0:
            ax.scatter(x1[i,0],x1[i,1],marker='v',alpha =1,color = 'b', s=12, label = 'Alaska')
            # plt.plot(x1[i,0],x1[i,1],'bv',markersize=5)
        else:
            ax.scatter(x1[i,0],x1[i,1],marker='o',alpha =1,color = 'r', s=12, label = 'Canada')
            # plt.plot(x1[i,0],x1[i,1],'ro',markersize=5)
    

    plt.title('Graph of GDA')
    plt.xlabel('X0', color='#1C2833')
    plt.ylabel('X1', color='#1C2833')
    plt.show()



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
    print(u0)
    print(u1)
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
    print(sigma)
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
    # print(coeff)
    plot_graph(y,x)
    plot_linear(y,x,coeff,const)
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
    print("u0",u0)
    print("u1",u1)
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
    print("sigma0",sigma0)
    print("sigma1",sigma1)
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
    plot_quadratic(c4,c3,c2,c1,y,x) 
    


def normalise(x):
    n,m = np.shape(x)
    # print(x)
    x[0][0:m] = x[0][0:m]- np.mean(x[0][0:m])
    x[0][0:m] = x[0][0:m]/ np.std(x[0][0:m])
    x[1][0:m] = x[1][0:m]- np.mean(x[1][0:m])
    x[1][0:m] = x[1][0:m]/ np.std(x[1][0:m])
    # print(x)

def main():
    x = np.genfromtxt('q4x.dat')
    # print(x)
    # y = np.genfromtxt('q4y.dat')
    # print(y)
    m,n = np.shape(x)
    y = np.zeros((m,1))
    fy = open('q4y.dat')
    lines = fy.readlines()
    i=0
    for line in lines:
        # print(line)
        if line == "Alaska\n":
            y[i][0]=0
        else:
            y[i][0]=1
        i+=1
    
    x=np.transpose(x)
    normalise(x)
    
    n,m = np.shape(x)

    # x1 = np.ones((1,m))
    # x = np.vstack((x1,x))
    # print(x)
    # print(y)

    general_gda(y,x)


    


main()
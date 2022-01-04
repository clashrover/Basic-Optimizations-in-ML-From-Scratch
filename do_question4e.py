
# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 
import math

import os
import sys

from util import general_gda, normalise

def plot_quadratic(c4,c3,c2,c1,y1,x1,u,v,output_dir):
    m,n = np.shape(x1)
    # print(x)
    # print(s1,m1,s2,m2)

    lpx = np.linspace(-50,50,1000)
    lpy = np.linspace(-50,50,1000)
    X, Y = np.meshgrid(lpx, lpy)
    # X = X
    # print(X)
    # X = X + m1
    # print(X)
    # Y = Y*s2
    # Y = Y + m2
    
    F = c4[0][0]*X**2 + (c4[1][0] + c4[0][1])*X*Y + c4[1][1]*Y**2 + c3[0][0]*X + c3[1][0]*Y + c2+c1

    fig,ax = plt.subplots()
    ax.contour(X*v[0]+u[0], Y*v[1]+u[1], F, levels=[0]) # take level set corresponding to 0
    
    for i in range(m):
        if y1[i][0]==0:
            ax.scatter(x1[i,0]*v[0]+u[0],x1[i,1]*v[1]+u[1],marker='v',alpha =1,color = 'b', s=12, label = 'Alaska')
            # plt.plot(x1[i,0],x1[i,1],'bv',markersize=5)
        else:
            ax.scatter(x1[i,0]*v[0]+u[0],x1[i,1]*v[1]+u[1],marker='o',alpha =1,color = 'r', s=12, label = 'Canada')
            # plt.plot(x1[i,0],x1[i,1],'ro',markersize=5)
    
    ax.set_xlim(40,200)
    ax.set_ylim(250,600)
    plt.title('Graph of GDA')
    plt.xlabel('X0', color='#1C2833')
    plt.ylabel('X1', color='#1C2833')
    plt.savefig(output_dir)


def main():
    
    inputx = os.path.join(sys.argv[1], 'q4x.dat')
    inputy = os.path.join(sys.argv[1], 'q4y.dat')
    x = np.genfromtxt(inputx)
    # x2 = np.copy(x)
    # print(np.shape(x))
    # y = np.genfromtxt('q4y.dat')
    # print(y)
    m,n = np.shape(x)
    # x2 = np.copy(x)
    y = np.zeros((m,1))
    fy = open(inputy)
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
    u=[]
    v=[]
    for i in range(n):
        u.append(np.mean(x[i][0:m]))
        v.append(np.std(x[i][0:m]))
        x[i][0:m] = (x[i][0:m]- np.mean(x[i][0:m]))/np.std(x[i][0:m])
    
    n,m = np.shape(x)

    # x1 = np.ones((1,m))
    # x = np.vstack((x1,x))
    # print(x)
    # print(y)
    # x=np.transpose(x)
    u0, u1 , sigma0, sigma1,c4,c3,c2,c1 = general_gda(y,x)
    x=np.transpose(x)
    output_dir = os.path.join(sys.argv[2], '4e.png')
    plot_quadratic(c4,c3,c2,c1,y,x,u,v,output_dir)


    


main()

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 
import math

import os
import sys

from util import gda, normalise
   



def main():
    inputx = os.path.join(sys.argv[1], 'q4x.dat')
    inputy = os.path.join(sys.argv[1], 'q4y.dat')
    x = np.genfromtxt(inputx)
    # print(np.shape(x))
    # y = np.genfromtxt('q4y.dat')
    # print(y)
    m,n = np.shape(x)
    x2 = np.copy(x)
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
    # normalise(x)
    n,m = np.shape(x)
    # print(x)
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
    u0, u1 , sigma, const, coeff = gda(y,x)

    output_dir = os.path.join(sys.argv[2], '4c.png')
    x=np.transpose(x)
    # plot_linear_gda(y,x,coeff,const, output_dir )

    m,n = np.shape(x)
    # print(x)
    for i in range(m):
        if y[i][0]==0:
            plt.plot(x[i,0]*v[0] + u[0],x[i,1]*v[1]+u[1],'bv',markersize=5)
        else:
            plt.plot(x[i,0]*v[0]+u[0],x[i,1]*v[1]+u[1],'ro',markersize=5)
    
    lpx = np.linspace(-1,2,100)
    lpy =  (coeff[0][0]*lpx)
    lpy = lpy + const
    lpy = lpy/ coeff[1][0]
    lpy = lpy*-1 

    plt.plot(lpx*v[0]+u[0],lpy*v[1]+u[1],'-g',label = 'GDA')
    plt.title('Graph of GDA')
    plt.xlabel('X1', color='#1C2833')
    plt.ylabel('X2', color='#1C2833')
    plt.savefig(output_dir)
    


main()
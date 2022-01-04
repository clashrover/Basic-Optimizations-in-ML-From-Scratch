
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 

import math
import sys
import os
 
from util import *

def plot_linearlog(y,x,theta,output_dir):
    m,n = np.shape(x)
    # print(x)
    for i in range(m):
        if y[i][0]==0:
            plt.plot(x[i,0],x[i,1],'bv',markersize=5)
        else:
            plt.plot(x[i,0],x[i,1],'ro',markersize=5)
    
    lpx = np.linspace(2,8,100)
    lpy =  (theta[1][0]*lpx)
    lpy = lpy + theta[0][0]
    lpy = lpy/ theta[2][0]
    lpy = lpy*-1 

    plt.plot(lpx,lpy,'-g',label = 'Logistic Regression')
    plt.title('Graph of Logistic Regression')
    plt.xlabel('X1', color='#1C2833')
    plt.ylabel('X2', color='#1C2833')
    # plt.show()
    plt.savefig(output_dir)


def main():
    inputx = os.path.join(sys.argv[1], 'logisticX.csv')
    inputy = os.path.join(sys.argv[1], 'logisticY.csv')
    output_dir = os.path.join(sys.argv[2], '3a.png')
    with open(inputx) as fx,open(inputy) as fy:
        lines = (line for line in fx if not line.startswith('#'))
        x = np.loadtxt(lines, delimiter=',')
        # f1 = np.transpose(f1)
        lines = (line for line in fy if not line.startswith('#'))
        y = np.loadtxt(lines, delimiter=',')
        # f2 = np.transpose(f2)
        x2 = np.copy(x)
        # print(np.shape(x2))
        # x2= np.transpose(x2)
        # print(np.shape(x1))
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
        lim = 0.000001
        k=0.1
        theta = logistic_reg(y,x,k,lim)
        plot_linearlog(y,x2,theta,output_dir)
    


main()
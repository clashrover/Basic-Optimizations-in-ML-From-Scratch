import os
import sys

from util import linear_regression, normalise


import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 


def main():
    inputx = os.path.join(sys.argv[1], 'linearX.csv')
    inputy = os.path.join(sys.argv[1], 'linearY.csv')
    output_dir = os.path.join(sys.argv[2], '1b.png')
    with open(inputx) as fx,open(inputy) as fy:
        lines = (line for line in fx if not line.startswith('#'))
        x = np.loadtxt(lines, delimiter=',')
        
        lines = (line for line in fy if not line.startswith('#'))
        y = np.loadtxt(lines, delimiter=',')
        
        m = np.size(x)
        
        x1 = np.zeros((1,m))
        x1[0,0:m] = x               #converting x to 1*m matrix

        x3 = np.copy(x1)
        x1=np.transpose(x1)
        av = np.mean(x1,axis=0)
        std = np.std(x1,axis=0)
        av = av[0]
        std=std[0]
        x1 = (x1-np.mean(x1,axis=0))/np.std(x1,axis=0) #normalise
        x1=np.transpose(x1)
        # x1 = normalise(x1)               #normalise
        # print(x1)
        x2 = np.ones((1,m))         #append a row with 1s
        x = np.vstack((x2,x1))
        
        y1 = np.zeros((1,m))
        y1[0][0:m]=y
        y1=np.transpose(y1)
        y=y1
        
        k=0.1
        lim, theta, theta0_list, theta1_list, error_list = linear_regression(y,x,k)
        
        x=np.transpose(x)           #some preprocessing for printing
        m,n = np.shape(x)
        x3 = np.transpose(x3)       #some preprocessing for printing
        plt.plot(((x[0:m,1]*std)+av),y[0:m,0],'bo',markersize=1)  
        
        lpx = np.linspace(-2,5,100)
        lpy = theta[0][0]+ theta[1][0]*lpx
        plt.plot(lpx*std+av,lpy,'-g',label = 'Linear Regression')
        
        # plt.plot((x_vals*std + av), y_vals, '-g')
        plt.title('Graph of Linear Regression')
        plt.xlabel('Acidity (X)', color='#1C2833')
        plt.ylabel('Density (Y)', color='#1C2833')
        plt.savefig(output_dir)

main()


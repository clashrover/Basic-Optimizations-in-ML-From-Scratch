
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

from util import general_gda, normalise


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
    normalise(x)
    
    n,m = np.shape(x)

    # x1 = np.ones((1,m))
    # x = np.vstack((x1,x))
    # print(x)
    # print(y)
    # x=np.transpose(x)
    u0, u1 , sigma0, sigma1,c4,c3,c2,c1 = general_gda(y,x)

    output_dir = os.path.join(sys.argv[2], '4d.txt')
    outf = open(output_dir,'w')
    print("u0 =\n", u0,
    "\nu1 = \n", u1, "\n"
    "Sigma0:\n", sigma0,
    "\nSigma1:\n", sigma1, file=outf)


    


main()
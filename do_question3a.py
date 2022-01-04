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


def main():
    inputx = os.path.join(sys.argv[1], 'logisticX.csv')
    inputy = os.path.join(sys.argv[1], 'logisticY.csv')
    output_dir = os.path.join(sys.argv[2], '3a.txt')
    with open(inputx) as fx,open(inputy) as fy:
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
        lim = 0.000001
        k=0.1
        theta = logistic_reg(y,x,k,lim)
        outf = open(output_dir,'w')
        print("For Logistic Reg learning rate =", k,
        "\nConvergence lim:", lim, "\n"
        "Theta:\n", theta, file=outf)
    


main()
import os
import sys

from util import sgd, generate,error


import matplotlib
matplotlib.use('Agg')

import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 
from sampling import generate



def main():
    # run for sampling and sgd uncomment
    x, y = generate()
    # print(x)
    # print(y)
    r1=1
    avg1 = 500
    l1= 0.0001

    r2=100
    avg2 = 10000
    l2=0.0001

    r3=10000
    avg3=10000
    l3=.0001

    r4 = 1000000
    avg4 = 1000
    l4=0.0001
    theta1, theta_list1 = sgd(y,x,r1,0.001,avg1,l1)
    
    theta2, theta_list2 = sgd(y,x,r2,0.001,avg2,l2)
    
    theta3, theta_list3 = sgd(y,x,r3,0.001,avg3,l3)

    theta4, theta_list4 = sgd(y,x,r4,0.001,avg4,l4)

    input = os.path.join(sys.argv[1], 'q2test.csv')
    with open(input) as f:
        lines = (line for line in f if not line.startswith('#'))
        f = np.loadtxt(lines, delimiter=',', skiprows=1)
        f = np.transpose(f)
        # print(r)
        x, y = np.split(f,[2])
        # x = np.transpose(x)
        y = np.transpose(y)
        r,c = np.shape(x)
        z = np.ones((1,c))
        x = np.vstack((z,x))
        
        x = np.transpose(x)
        error1  = error(y,x,theta1)
        error2  = error(y,x,theta2)
        error3  = error(y,x,theta3)
        error4  = error(y,x,theta4)
        theta5 = np.zeros((3,1))
        theta5[0][0]=3
        theta5[1][0]=1
        theta5[2][0]=2
        
        error5 = error(y,x,theta5)

        out_dir = os.path.join(sys.argv[2], '2c.txt')
        outf = open(out_dir,'w')
        print("error1 =", error1,
            "\nerror2 =", error2,
            "\nerror3 =", error3,
            "\nerror4 =", error4,
            "\n original hypo error5 =", error5,
            file=outf)
        


main()
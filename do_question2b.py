import os
import sys

from util import sgd, generate


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
    # print(theta1)
    theta2, theta_list2 = sgd(y,x,r2,0.001,avg2,l2)
    # print(theta2)
    theta3, theta_list3 = sgd(y,x,r3,0.001,avg3,l3)
    # print(theta3)
    theta4, theta_list4 = sgd(y,x,r4,0.001,avg4,l4)
    # # print(theta4)
    out_dir = os.path.join(sys.argv[2], '2b.txt')
    outf = open(out_dir,'w')
    print("theta for batch_size 1 =\n", theta1,
        "\ntheta for batch_size 2 =\n", theta2,
        "\ntheta for batch_size 3 =\n", theta3,
        "\ntheta for batch_size 4 =\n", theta4,
        file=outf)
    

main()
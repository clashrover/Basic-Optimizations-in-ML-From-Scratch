import os
import sys

from util import sgd, generate,error


# import matplotlib
# matplotlib.use('Agg')

import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 
from sampling import generate


def plot_3d(theta_list,outp):
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(0, 3.5)
    ax.set_ylim3d(0,1.5)
    ax.set_zlim3d(0,2.5)
    fps = 50 # frame per sec
    frn = len(theta_list) # frame number of the animation

    x=[]
    y=[]
    z=[]
    for i in range(len(theta_list)):
        x.append(theta_list[i][0])
        y.append(theta_list[i][1])
        z.append(theta_list[i][2])

    ax.scatter(x,y,z,marker='o',alpha =1,color = 'r', s=2)


    plt.savefig(outp) 



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

    # theta4, theta_list4 = sgd(y,x,r4,0.001,avg4,l4)

    input1 = os.path.join(sys.argv[2], '2d1.png')
    input2 = os.path.join(sys.argv[2], '2d2.png')
    input3 = os.path.join(sys.argv[2], '2d3.png')
    # input4 = os.path.join(sys.argv[2], '2d4.png')
    plot_3d(theta_list1,input1)
    plot_3d(theta_list2,input2)
    plot_3d(theta_list3,input3)
    # plot_3d(theta_list4,input4)

    

main()
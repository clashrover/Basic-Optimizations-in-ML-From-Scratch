import os
import sys

from util import linear_regression, error, grad



import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 


def contour_animation(y,x,theta0_list,theta1_list, error_list, output_dir):
    #same as last part
    # Make data fo the contour graph
    th1 = np.arange(0, 2, 0.05)
    th0 = np.arange(-1, 1, 0.05)
    th1, th0 = np.meshgrid(th1, th0)
    z1 = grad(th1,th0,y,x)

    fig, ax = plt.subplots()
    CS = ax.contour(th1, th0, z1)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Contour plot of Error')
    ax.set_xlabel('Θ0')
    ax.set_ylabel('Θ1')

    l = len(theta1_list)
    t0 = np.zeros((l))
    t0[0:l]=theta0_list
    t1 = np.zeros((l))
    t1[0:l]=theta1_list
    e = np.zeros((l))
    e[0:l]=error_list

    # ax.scatter(x1,y1,z2,marker='o',alpha =1,color = 'r', s=3)


    fps = 5 # frame per sec
    frn = 1000 # frame number of the animation

    def update_plot(frame_number, zarray, plot):
        plot[0].remove()
        plot[0] = ax.scatter(t0[0:frame_number],t1[0:frame_number],marker='o',alpha =1,color = 'r', s=12)


    plot = [ax.scatter(t0,t1,marker='o',alpha =1,color = 'r', s=12)]
    ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(e, plot), interval=1000/fps)

    # plt.show()

    ani.save(output_dir,writer='imagemagick',fps=fps)


def draw(k,f):
    inputx = os.path.join(sys.argv[1], 'linearX.csv')
    inputy = os.path.join(sys.argv[1], 'linearY.csv')
    output_dir = os.path.join(sys.argv[2], f)
    with open(inputx) as fx,open(inputy) as fy:
        lines = (line for line in fx if not line.startswith('#'))
        x = np.loadtxt(lines, delimiter=',')
        
        lines = (line for line in fy if not line.startswith('#'))
        y = np.loadtxt(lines, delimiter=',')
        
        x = np.transpose(x)
        x1 = (x-np.mean(x))
        x1 = x1/np.std(x)
        x=x1
        m = np.size(x)
        
        x1 = np.zeros((1,m))
        x1[0,0:m] = x
        
        x2 = np.ones((1,m))
        x = np.vstack((x2,x1))
        
        y1 = np.zeros((1,m))
        y1[0][0:m]=y
        y1=np.transpose(y1)
        y=y1
        
        lim, theta, theta0_list, theta1_list, error_list = linear_regression(y,x,k)
        
        x=np.transpose(x)
        
        contour_animation(y,x,theta0_list,theta1_list,error_list, output_dir)

def main():
    k1 = 0.0001
    k2 = 0.025
    k3 = 0.1
    f1 = "a.gif"
    f2 = "b.gif"
    f3 = "c.gif"
    draw(k1,f1) # draw contours for different eta
    draw(k2,f2)
    draw(k3,f3)

main()

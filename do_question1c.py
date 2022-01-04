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


def plot_3d_animation(y,x,theta0_list,theta1_list, error_list,output_dir):
    fig = plt.figure(figsize=(8,6))                             #form the figure
    ax = fig.add_subplot(111, projection='3d')                  #set ax
    th1 = np.arange(-0.5, 2.5, 0.05)                            #form arrays in range -0.5 to 2.5 with step size 0.05
    th0 = np.arange(-1, 1, 0.05)
    th1, th0 = np.meshgrid(th1, th0)                            # create mesh grid
    z1 = grad(th1,th0,y,x)                                      # for each point on grid store z

    surf = ax.plot_surface(th1, th0, z1, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none',alpha=0.5) #surface plot
    ax.set_xlabel('Θ0')
    ax.set_ylabel('Θ1')
    ax.set_zlabel('JΘ')
    plt.title('Graph of JΘ')
    # # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.3, aspect=5)
    l = len(theta1_list)
    t0 = np.zeros((l))
    t0[0:l]=theta0_list
    t1 = np.zeros((l))
    t1[0:l]=theta1_list
    e = np.zeros((l))
    e[0:l]=error_list

    fps = 2 # frame per sec
    frn = 100 # frame number of the animation

    def update_plot(frame_number, zarray, plot):
        plot[0].remove()
        plot[0] = ax.scatter(t0[0:frame_number],t1[0:frame_number],e[0:frame_number],marker='o',alpha =1,color = 'r', s=12) #plot points till frame number


    plot = [ax.scatter(t0,t1,e,marker='o',alpha =1,color = 'r', s=12)]
    ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(e, plot), interval=1000/fps) #call animation function


    ani.save(output_dir,writer='imagemagick',fps=fps)




def main():
    inputx = os.path.join(sys.argv[1], 'linearX.csv')
    inputy = os.path.join(sys.argv[1], 'linearY.csv')
    output_dir = os.path.join(sys.argv[2], '1c.gif')
    with open(inputx) as fx,open(inputy) as fy:
        lines = (line for line in fx if not line.startswith('#'))
        x = np.loadtxt(lines, delimiter=',')
        
        lines = (line for line in fy if not line.startswith('#'))
        y = np.loadtxt(lines, delimiter=',')
        
        x = np.transpose(x)
        x1 = (x-np.mean(x))/np.std(x)
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
        
        k=0.1
        lim, theta, theta0_list, theta1_list, error_list = linear_regression(y,x,k)
        
        x=np.transpose(x)
        
        
        plot_3d_animation(y,x,theta0_list,theta1_list,error_list,output_dir)

main()


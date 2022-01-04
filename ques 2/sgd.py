import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 
from sampling import generate


# function to calculate gradient of JΘ
def gradient(y,x,theta):
    m,n = np.shape(x)
    r = np.matmul(np.transpose(x),x)
    r = np.matmul(r,theta)
    d = np.matmul(np.transpose(x),y)
    g = r-d
    g = g/m

    return g

# convergence test
def converge(e1, e,lim):
    return abs(e1-e) < lim

def error(y,x,theta):
    m,n = np.shape(x)
    r = np.matmul(x,theta)- y
    e = np.matmul(np.transpose(r),r)
    e = e/(2*m)
    # print(e[0][0])
    return e[0][0]

def shuffle(y,x):
    """
    To shuffle the data set
    """
    shuf  = np.vstack((np.transpose(y),x))
    shuf = np.transpose(shuf)
    np.random.shuffle(shuf)
    shuf = np.transpose(shuf)
    y,x = np.split(shuf,[1])
    y=np.transpose(y)
    return y,x

def plot_3d(theta_list):
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
    def update_plot(frame_number, zarray, plot):
        plot[0].remove()
        t1 = theta_list[frame_number]
        x.append(t1[0][0])
        y.append(t1[1][0])
        z.append(t1[2][0])
        plot[0] = ax.scatter(x,y,z,marker='o',alpha =1,color = 'r', s=2)


    plot = [ax.scatter(theta_list[0][0],theta_list[0][1],theta_list[0][2],marker='o',alpha =1,color = 'r', s=12)]
    ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(theta_list[0:][2], plot), interval=1000/fps)

    plt.show()

    # fn = 'GD 3D_0.05_e(-8)'
    # ani.save(fn+'.gif',writer='imagemagick',fps=fps)
    


#-------------------------------------------------------------------------------------
# Stochastic Gradient Descent


def sgd(y,x,k):
    # x, y = generate()
    n,m = np.shape(x)
    # form the x matrix with each row as xi that is a matrix of size 2*1
    theta = np.zeros((n,1))
    l =0.001
    
    t=0 #time                 
    

    r=10000
    epoch=0

    e_old = 0
    e =0
    avg=10000

    epoch_limit = 10000
    y,x = shuffle(y,x)
    x=np.transpose(x)    #design matrix

    theta_list = []   #for plotting

    while True:
        bln=False
        for b in range(math.floor(m/r)):
            p = gradient(y[b*r:r*(b+1)],x[b*r:r*(b+1)],theta)
            theta1 = theta - (k*p)
            e+= error(y[b*r:r*(b+1)],x[b*r:r*(b+1)],theta1)
            t+=1
            theta = theta1
            if(t%15 == 0):
                theta_list.append(theta) 
            
            if t%avg == 0:
                e=e/avg
                # print(e,e_old)
                if converge(e,e_old,l):
                    # print("-----")
                    bln=True
                    break
                print(e)
                e_old = e
                e=0
        
        if epoch > epoch_limit:
            break
        if bln:
            break
        epoch+=1
        
    # print("For batch size:",r)
    # print("Convergence limit:",l)
    # if epoch>epoch_limit:
    #     print("-----------time out-------------")
    # print("Avg over:",avg)
    # print(theta)
    # print("Final error :",e_old)
    # print("Iterations:",t)
    theta_list.append(theta)
    print(theta)
    plot_3d(theta_list)

    # "Some results"



    # For batch size: 1
    # Convergence limit: 1e-4
    # Avg over: 10000
    # [[2.9838448 ]
    #  [0.97211639]
    #  [1.97694243]]
    # Final error : 0.9632295750134213
    # Iterations: 580000

    # For batch size: 100
    # Convergence limit: 0.0001
    # Avg over: 10000
    # [[3.00081062]
    #  [0.99871352]
    #  [1.99618362]]
    # Final error : 1.001252073978551
    # Iterations: 40000

    # For batch size: 10000
    # Convergence limit: 0.0001
    # Avg over: 10000
    # [[2.99810016]
    #  [1.00133524]
    #  [2.0002265 ]]
    # Final error : 1.000933790291092
    # Iterations: 40000


    #Stochastic Gradient Descent
    #For batch size: 1000000
    #Convergence limit: 0.0001
    #-----------time out-------------
    #Avg over: 1000000
    #[[2.8153132 ]
    # [1.04045065]
    # [1.9879714 ]]
    #Final error : 0
    #Iterations: 10002

def main():
    # run for sampling and sgd uncomment
    x, y = generate()
    # print(x)
    # print(y)
    sgd(y,x,0.001)

    #for q2test.csv
    # with open('q2test.csv') as f:
    #     lines = (line for line in f if not line.startswith('#'))
    #     f = np.loadtxt(lines, delimiter=',', skiprows=1)
    #     f = np.transpose(f)
    #     # print(r)
    #     x, y = np.split(f,[2])
    #     # x = np.transpose(x)
    #     y = np.transpose(y)
    #     r,c = np.shape(x)
    #     z = np.ones((1,c))
    #     x = np.vstack((z,x))
    #     sgd(y,x,0.001)



main()

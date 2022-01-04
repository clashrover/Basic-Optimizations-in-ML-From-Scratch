import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 

def plot_linear(y,x,theta):
    m,n = np.shape(x)
    plt.plot(x[0:m,1],y[0:m,0],'ro',markersize=1)
    lpx = np.linspace(-2,5,100)
    lpy = theta[0][0]+ theta[1][0]*lpx
    plt.plot(lpx,lpy,'-g',label = 'Linear Regression')
    plt.title('Graph of Linear Regression')
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.show()


def grad(t1,t0,y,x):
    z = np.zeros(np.shape(t1))
    r,c = np.shape(t1)
    for i in range(r):
        for j in range(c):
            th = np.zeros((2,1))
            th[0][0]=t1[i][j]
            th[1][0]=t0[i][j]
            z[i][j] = error(y,x,th)
    return z

def plot_3d_animation(y,x,theta0_list,theta1_list, error_list):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    th1 = np.arange(-0.5, 2.5, 0.05)
    th0 = np.arange(-1, 1, 0.05)
    th1, th0 = np.meshgrid(th1, th0)
    z1 = grad(th1,th0,y,x)

    surf = ax.plot_surface(th1, th0, z1, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none',alpha=0.5)
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
        plot[0] = ax.scatter(t0[0:frame_number],t1[0:frame_number],e[0:frame_number],marker='o',alpha =1,color = 'r', s=12)


    plot = [ax.scatter(t0,t1,e,marker='o',alpha =1,color = 'r', s=12)]
    ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(e, plot), interval=1000/fps)

    # plt.show()

    # fn = 'GD 3D_0.05_e(-8)'
    # ani.save(fn+'.gif',writer='imagemagick',fps=fps)


    plt.show()

def contour_animation(y,x,theta0_list,theta1_list, error_list):

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

    plt.show()

    # fn = 'Contour_plot_0.001_e(-8)'
    # ani.save(fn+'.gif',writer='imagemagick',fps=fps)


# ---------------------------------------------------------------------
# calculate gradient of JΘ
def gradient(y,x,theta):
    m,n = np.shape(x)
    r = np.matmul(np.transpose(x),x)
    r = np.matmul(r,theta)
    d = np.matmul(np.transpose(x),y)
    g = r-d
    g = g/m
    return g

# convergence test
def converge(e1, e):
    return abs(e1-e) < 0.00000001

def error(y,x,theta):
    m,n = np.shape(x)
    r = np.matmul(x,theta)- y
    e = np.matmul(np.transpose(r),r)
    e = e/(2*m)
    return e[0][0]
#-------------------------------------------------------------------------------------
# linear regression using gd

def linear_regression(y,x,k):

    
    n,m = np.shape(x)
    
    theta = np.zeros((n,1))
    
    
    t=0 
    
    x=np.transpose(x)           
    
    e= error(y,x,theta)
    t=0
    
    theta0_list = [theta[0][0]]
    theta1_list = [theta[1][0]]
    error_list =[e]

    while True:
        # print(e)
        p = gradient(y,x,theta)
        theta1 = theta - (k*p)
        theta0_list.append(theta[0][0])
        theta1_list.append(theta[1][0])
        
        e1 = error(y,x,theta1)
        error_list.append(e1)

        t+=1
        if converge(e1, e):
            theta = theta1
            break
        theta = theta1
        e = e1

    
    
    plot_linear(y,x,theta)
    plot_3d_animation(y,x,theta0_list,theta1_list,error_list)
    contour_animation(y,x,theta0_list,theta1_list,error_list)
    

def main():
    with open('linearX.csv') as fx,open('linearY.csv') as fy:
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
        
        linear_regression(y,x,0.1)

main()


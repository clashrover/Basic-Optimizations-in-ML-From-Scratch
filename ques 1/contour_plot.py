import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 


# function to calculate gradient of JΘ
def summation(y,x,theta):
    m,b,a = np.shape(y) 
    # print(m)
    r,c = np.shape(x[0])
    sum = np.zeros((r,c))
    for i in range(m):
        p1 = (y[i] - np.matmul(np.transpose(theta),x[i]))
        sum+= -1*p1[0,0]*x[i]
    sum = sum/m
    return sum

# convergence test
def converge(e1, e):
    return abs(e1-e) < 0.00000001

def error(y,x,theta):
    m,b,a = np.shape(y)
    # print(m)
    e = 0
    for i in range(m):
        p1 = (y[i] - np.matmul(np.transpose(theta),x[i]))
        e+= p1[0,0]*p1[0,0]
    e=e/(2*m)
    return e

def grad(t1,t0,y,x):
    z = np.zeros(np.shape(t1))
    r,c = np.shape(t1)
    for i in range(r):
        for j in range(c):
            th = np.zeros((2,1))
            th[0][0]=t1[i][j]
            th[1][0]=t0[i][j]
            z[i][j] = error(y,x,th)
            # print(z[i][j])
    return z
#-------------------------------------------------------------------------------------



# Main program

x = np.genfromtxt('linearX.csv')    #take input
y = np.genfromtxt('linearY.csv')    #take input
m = np.size(x)                      #number of training example
n = np.size(x[0])                   #number of features
x = (x - np.mean(x))/ np.std(x)     #standardization of x's: mean 0 and variance is 1
x1=x.copy()
y1=y.copy()


# form the x matrix with each row as xi that is a matrix of size 2*1
theta = np.zeros((n+1,1))
xn  = np.ones((1,n+1,m))
xn[0:1,0:(n),0:(m+1)] = x
xn = np.transpose(xn)
x=xn

#form the y matrix with each row as yi that is a matrix of size 1*1
yn = np.zeros((1,n,m))
yn[0:1,0:n,0:(m+1)]=y
yn = np.transpose(yn)
y=yn

#print shapes to confirm formation
print(np.shape(x)) 
print(np.shape(y))
print(np.shape(theta)) 
t=0 #time
k = 0.001                         #learning rate

#setting 3d curve

xs =[]
ys =[]
zs =[]

# gradient descent algorithm
e= error(y,x,theta)
xs.append(theta[0][0])
ys.append(theta[1][0])
zs.append(e)
t=0
while True:
    # print(e)
    p = summation(y,x,theta)
    theta1 = theta - (k*p)
    t=t+1
    e1 = error(y,x,theta1)
    
    xs.append(theta1[0][0])
    ys.append(theta1[1][0])
    zs.append(e1)

    if converge(e1, e):
        theta = theta1
        break
    theta = theta1
    e = e1


print(t)
l = len(xs)
x1 = np.zeros(l)
y1 = np.zeros(l)
z2 = np.zeros(l)
x1[0:l] = xs
y1[0:l] = ys
z2[0:l] = zs



# Make data fo the contour graph
th1 = np.arange(-1, 1, 0.05)
th0 = np.arange(0, 2, 0.05)
th1, th0 = np.meshgrid(th1, th0)
z1 = grad(th1,th0,y,x)

fig, ax = plt.subplots()
CS = ax.contour(th1, th0, z1)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Contour plot of Error')

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)


# ax.scatter(x1,y1,z2,marker='o',alpha =1,color = 'r', s=3)


fps = 5 # frame per sec
frn = 1000 # frame number of the animation

def update_plot(frame_number, zarray, plot):
    plot[0].remove()
    plot[0] = ax.scatter(x1[0:frame_number],y1[0:frame_number],marker='o',alpha =1,color = 'r', s=12)


plot = [ax.scatter(x1,y1,marker='o',alpha =1,color = 'r', s=12)]
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(z2, plot), interval=1000/fps)

#plt.show()

fn = 'Contour_plot_0.001_e(-8)'
ani.save(fn+'.gif',writer='imagemagick',fps=fps)

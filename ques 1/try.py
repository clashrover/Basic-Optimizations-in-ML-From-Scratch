import numpy as np
print('numpy: '+np.version.full_version)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
import matplotlib
print('matplotlib: '+matplotlib.__version__)

N = 150 # Meshsize
fps = 10 # frame per sec
frn = 50 # frame number of the animation

x1 = np.ones(N+1)
y1 = np.ones(N+1)
z2 = x1+y1

def update_plot(frame_number, zarray, plot):
    plot[0].remove()
    plot[0] = ax.scatter(x1[0:frame_number],y1[0:frame_number],z2[0:frame_number],marker='o',alpha =1,color = 'r', s=frame_number)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot = [ax.scatter(x1,y1,z2,marker='o',alpha =1,color = 'r', s=3)]
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(z2, plot), interval=1000/fps)

plt.show()
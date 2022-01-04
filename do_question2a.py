import os
import sys

from util import generate



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
    # sgd(y,x)

    # #for q2test.csv
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
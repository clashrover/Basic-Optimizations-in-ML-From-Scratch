import numpy as np
import math

def generate():
    theta = np.array([[3],[1],[2]])
    # print(theta)
    sample_size = 1000000
    x = np.zeros((sample_size,3))
    # print(np.shape(x))
    x[0:sample_size,0] = np.ones(sample_size)
    mu=3.0
    sig = 4.0
    x[0:sample_size,1] = np.random.normal(mu, math.sqrt(sig),sample_size)
    mu=-1.0
    sig=4.0
    x[0:sample_size,2] = np.random.normal(mu, math.sqrt(sig),sample_size)
    x = np.transpose(x)
    # print(x)
    y = np.zeros((sample_size,1))
    # print(y)
    y[0:sample_size,0] = np.matmul(np.transpose(theta),x)
    # print(y)
    mu =0.0
    sig = 2
    normal = np.random.normal(mu,math.sqrt(sig),sample_size)
    # print(normal)
    y=np.transpose(y)
    y[0,0:sample_size] = y[0,0:sample_size] + normal
    # print(y)
    y = np.transpose(y)
    return (x,y)

# generate()
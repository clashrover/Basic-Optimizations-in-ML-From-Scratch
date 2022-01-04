import numpy as np
import sys
import os
 
from util import linear_regression, normalise

def main():
    inputx = os.path.join(sys.argv[1], 'linearX.csv')
    inputy = os.path.join(sys.argv[1], 'linearY.csv')   #open the files
    output_dir = os.path.join(sys.argv[2], '1a.txt')
    with open(inputx) as fx,open(inputy) as fy:
        lines = (line for line in fx if not line.startswith('#'))
        x = np.loadtxt(lines, delimiter=',')            #some preprocessing
        
        lines = (line for line in fy if not line.startswith('#'))
        y = np.loadtxt(lines, delimiter=',')
        
        # x = np.transpose(x)                             
        # x1 = (x-np.mean(x))
        # x1 = x1/np.std(x)
        # x=x1
        m = np.size(x)
        
        x1 = np.zeros((1,m))
        x1[0,0:m] = x        #converting list into 1*m size matrix

        normalise(x1)       # normalise the x
        x2 = np.ones((1,m))
        x = np.vstack((x2,x1))  #append a row with 1s. Now it is 2*m matrix 
        
        y1 = np.zeros((1,m))
        y1[0][0:m]=y            #convert y to m*1 matrix
        y1=np.transpose(y1)
        y=y1
        
        k=0.1
        lim, theta, theta0_list, theta1_list, error_list = linear_regression(y,x,k) #call linear reg
        
        outf = open(output_dir,'w')
        print("Learning Rate =", k,
        "\nStopping Rate:", lim, "\n"
        "Theta:\n", theta, file=outf)

main()


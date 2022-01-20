import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e2, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-1, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.7, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')
n = seq.shape[2]

for i in range(n-1):
    print(i)
    It = seq[:,:,i]
    It1 = seq[:,:,i+1]
    mask = SubtractDominantMotion(It, It1,threshold, num_iters, tolerance)
    
    print (mask.shape)
    if i == 30 or i == 60 or i == 90 or i == 120:
        plt.figure()
        plt.imshow(It, cmap='gray')
        for j in range(mask.shape[0]):
            for k in range(mask.shape[1]):
                if mask[j,k] == 1:
                    plt.scatter(k, j,s = 1, c = 'blue')
        plt.title('Frame %d'%i)
        plt.show()

#print(masks.shape)
    
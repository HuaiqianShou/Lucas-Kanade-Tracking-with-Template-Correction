import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import copy

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=2500, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=0.05, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
#template_threshold = args.template_threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

h = rect[2]-rect[0]+1
w = rect[3]-rect[1]+1

output = copy.deepcopy(rect)
output = np.array(output)
n = seq.shape[2]
#print(n)
for i in range(n-1):
    print(i)
    It = seq[:,:,i]
    It1 = seq[:,:,i+1]
    p = LucasKanade(It,It1,rect,threshold,num_iters)
    rect[0] = rect[0]+p[0]
    rect[1] = rect[1]+p[1]
    rect[2] = rect[2]+p[0]
    rect[3] = rect[3]+p[1]
    output = np.concatenate((output,rect))
    
    if i == 1 or i == 20 or i == 40 or i == 60 or i == 80:
        
        corner_left = rect[0]
        corner_right = rect[1]
        rect_track = patches.Rectangle((corner_left,corner_right),h,w,linewidth=1,edgecolor='r',facecolor='none')
        fig,ax = plt.subplots(1)
        ax.imshow(seq[:,:,i+1],cmap='gray')
        ax.add_patch(rect_track)
        plt.title('Frame %d'%i)
        plt.show()    
        
output = output.reshape(n,4)        
np.save('girlseqrects.npy',output)
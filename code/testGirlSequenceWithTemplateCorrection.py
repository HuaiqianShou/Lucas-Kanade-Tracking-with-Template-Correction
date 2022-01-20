import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import copy
from scipy.interpolate import RectBivariateSpline

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/girlseq.npy")
rect_normal = np.load("../code/girlseqrects.npy",allow_pickle=True)
rect = [280, 152, 330, 318]
rect_fix = copy.deepcopy(rect)
hf = rect[2]-rect[0]+1
wf = rect[3]-rect[1]+1

output = copy.deepcopy(rect)
output = np.array(output)
First = seq[:, :, 0]
n = seq.shape[2]

for i in range(n-1):
    print(i)
    It = seq[:,:,i]
    It1 = seq[:,:,i+1]

    pn = LucasKanade(It, It1, rect,threshold,num_iters, np.zeros(2))


    h = First.shape[0]
    w = First.shape[1]
    
    h1 = It1.shape[0]
    w1 = It1.shape[1]
    
    h = np.arange(h)   
    w = np.arange(w) 
    
    h1 = np.arange(h1)
    w1 = np.arange(w1)

    spline_It = RectBivariateSpline(h, w, First)
    spline_It1 = RectBivariateSpline(h1, w1,It1)
    
    
    x1_dr = rect[0]
    y1_dr = rect[1]
    x2_dr = rect[2]
    y2_dr = rect[3]

    x1 = rect_fix[0]
    y1 = rect_fix[1]
    x2 = rect_fix[2]
    y2 = rect_fix[3]   
    h_rec = x2 - x1 + 1
    w_rec = y2 - y1 + 1

    mesh_h, mesh_w = np.meshgrid(np.linspace(x1, x2, h_rec),np.linspace(y1, y2, w_rec)) 

    #print("mh:",mesh_h)
    flat_h =np.ndarray.flatten(mesh_h)
    flat_w =np.ndarray.flatten(mesh_w)
    
    
#    T = spline_It.ev(mesh_w,mesh_h)
    T = spline_It.ev(flat_w ,flat_h)

    

    I_x, I_y = np.gradient(It1)
        
    spline_I_x = RectBivariateSpline(h, w, I_x)
    
    
    spline_I_y = RectBivariateSpline(h, w, I_y)


    dp = np.array([[1], [1]])


    pc = copy.deepcopy(pn)
    j = 0
    while np.square(dp).sum() > threshold and j<=num_iters:

        j = j+1
        
        
        x1p = x1_dr + pc[0]
        x2p = x2_dr + pc[0]
        y1p = y1_dr + pc[1]
        y2p = y2_dr + pc[1]


        mesh_h_p, mesh_w_p = np.meshgrid(np.linspace(x1p, x2p, h_rec), np.linspace(y1p, y2p, w_rec))
        
        
        flat_mesh_h = np.ndarray.flatten(mesh_h_p)
        flat_mesh_w = np.ndarray.flatten(mesh_w_p)
        

        I_xp = spline_I_x.ev(flat_mesh_w , flat_mesh_h)
        I_yp = spline_I_y.ev(flat_mesh_w , flat_mesh_h)

        

        G_I_hello = np.stack((I_yp, I_xp), axis=-1)
  
        J = np.array([[1,0],[0,1]])

        H = (G_I_hello @ J).T @ (G_I_hello @ J)
        

        

        IW = spline_It1.ev(flat_mesh_w, flat_mesh_h)
        
        dp = np.linalg.inv(H) @ ((G_I_hello @ J).T) @ (T - IW)

        pc[0] = pc[0] + dp[0]
        pc[1] = pc[1] + dp[1]

    pn_s = pc
    

    if np.linalg.norm(pn_s - pn) <= template_threshold:
           
        rect[0] = rect[0] + pn_s[0]
        rect[1] = rect[1] + pn_s[1]
        rect[2] = rect[2] + pn_s[0]
        rect[3] = rect[3] + pn_s[1]
    else:
   
        rect[0] = rect[0] + pn[0]
        rect[1] = rect[1] + pn[1]
        rect[2] = rect[2] + pn[0]
        rect[3] = rect[3] + pn[1]
        
        

    output = np.concatenate((output,rect))

    if i == 1 or i == 20 or i == 40 or i == 60 or i == 80:
        
      
        corner_left = rect[0]
        corner_right = rect[1]
        corner_left_normal = rect_normal[i][0]
        corner_right_normal = rect_normal[i][1]
        rect_track_normal = patches.Rectangle((corner_left_normal,corner_right_normal),hf,wf,linewidth=1,edgecolor='blue',facecolor='none')
        rect_track_correction = patches.Rectangle((corner_left,corner_right),hf,wf,linewidth=1,edgecolor='r',facecolor='none')
        fig,ax = plt.subplots(1)
        ax.imshow(seq[:,:,i+1],cmap='gray')
        ax.add_patch(rect_track_normal)
        ax.add_patch(rect_track_correction)
        plt.title('Frame %d'%i)
        plt.show()   

 
output = output.reshape(n,4)  
np.save('girlseqrects-wcrt.npy',output)
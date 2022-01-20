import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.eye(3)
    h = It.shape[0]
    w = It.shape[1]
    height = It.shape[0]
    width = It.shape[1]
    
    h1 = It1.shape[0]
    w1 = It1.shape[1]
    
    h = np.arange(h)   
    w = np.arange(w) 
    
    h1 = np.arange(h1)
    w1 = np.arange(w1)
    
    spline_It = RectBivariateSpline(h, w, It)
    spline_It1 = RectBivariateSpline(h1, w1, It1)
    x1 = 0
    y1 = 0
    x2 = It.shape[1]
    y2 = It.shape[0]
    h_rec = x2 - x1
    w_rec = y2 - y1
    
    
    mesh_h, mesh_w = np.meshgrid(np.linspace(x1, x2, h_rec),np.linspace(y1, y2, w_rec)) 
    flat_h =mesh_h.flatten()
    flat_w =mesh_w.flatten()
    
    T = spline_It.ev(flat_w,flat_h) 
    
    I_x, I_y = np.gradient(It1)
    spline_I_x = RectBivariateSpline(h, w, I_x)
    spline_I_y = RectBivariateSpline(h, w, I_y)
    
    dp = np.zeros(6)
    p = np.zeros(6)

    i = 0
    while np.square(dp).sum() > threshold and i < num_iters:
        i = i+1
        p1 = p[0]
        p2 = p[1]
        p3 = p[2]
        p4 = p[3]
        p5 = p[4]
        p6 = p[5]
    
        x1p = (1+p1)*x1 + p2*y1 + p3
        y1p = p4*x1 + (1+p5)*y1 + p6
        x2p = (1+p1)*x2 + p2*y2 + p3
        y2p = p4*x2 + (1+p5)*y2 + p6
        
        #print(x1p)
        
        
        mesh_h_p, mesh_w_p = np.meshgrid(np.linspace(x1p, x2p, h_rec), np.linspace(y1p, y2p, w_rec))
        
        flat_mesh_h = mesh_h_p.flatten()
        flat_mesh_w = mesh_w_p.flatten()
        
        
        
        I_xp = spline_I_x.ev(flat_mesh_w, flat_mesh_h)
        I_yp = spline_I_y.ev(flat_mesh_w, flat_mesh_h)
        
        
        


        #print(I_yp.shape)

        G_I = np.stack((I_yp, I_xp), axis=-1)
        #print(I.shape)
        

        d = np.zeros((height*width, 6))
   
        for i in range(height):
            for j in range(width):
                J = np.array([[j, 0, i, 0, 1, 0],[0, j, 0, i, 0, 1]]) 
                d[width*i+j] = (G_I[width*i+j]) @ J
        

        H = d.T @ d
        
        IW = spline_It1.ev(flat_mesh_w, flat_mesh_h)
        dp = np.linalg.inv(H) @ (d.T) @ (T - IW)
        

        p[0] = p1 + dp[0]
        p[1] = p2 + dp[1]
        p[2] = p3 + dp[2]
        p[3] = p4 + dp[3]
        p[4] = p5 + dp[4]
        p[5] = p6 + dp[5]

    M =  np.array([[1 + p[0], p[1], p[2]], [p[3], 1 + p[4], p[5]],[0, 0, 1]])   
    return M

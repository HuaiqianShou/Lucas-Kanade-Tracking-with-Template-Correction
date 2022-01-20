import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """

    # Put your implementation here
    h = It.shape[0]
    w = It.shape[1]
    
    h1 = It1.shape[0]
    w1 = It1.shape[1]
    
    h = np.arange(h)   
    w = np.arange(w) 
    
    h1 = np.arange(h1)
    w1 = np.arange(w1)
    
    spline_It = RectBivariateSpline(h, w, It)
    spline_It1 = RectBivariateSpline(h1, w1, It1)
    
    #print("spIT:",spline_It)
    
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]
    h_rec = x2 - x1 + 1
    w_rec = y2 - y1 + 1
    #print("hehe:",x1,x2)
    mesh_h, mesh_w = np.meshgrid(np.linspace(x1, x2, h_rec),np.linspace(y1, y2, w_rec)) 

    #print("mh:",mesh_h)
#    mesh_h = np.array(mesh_h)
    flat_h =mesh_h.flatten()
    flat_w =mesh_w.flatten()
    
    
#    T = spline_It.ev(mesh_w,mesh_h)
    T = spline_It.ev(flat_w ,flat_h)
    
    
    I_x, I_y = np.gradient(It1)
        
    spline_I_x = RectBivariateSpline(h, w, I_x)
    
    
    spline_I_y = RectBivariateSpline(h, w, I_y)
    

    dp = [[10], [10]] 
    
    
    
    i = 0
    while np.square(dp).sum() > threshold and i<=num_iters:
            
        i = i+1
        x1p = x1+p0[0]
        x2p = x2+p0[0]
        y1p = y1 + p0[1]
        y2p = y2 + p0[1]   
#        I_xp, I_yp = np.gradient(It)
        mesh_h_p, mesh_w_p = np.meshgrid(np.linspace(x1p, x2p, h_rec), np.linspace(y1p, y2p, w_rec))
        
        
        flat_mesh_h = mesh_h_p.flatten()
        flat_mesh_w = mesh_w_p.flatten()
        
#        I_xp = spline_I_x.ev(mesh_w_p, mesh_h_p)
#        I_yp = spline_I_y.ev(mesh_w_p, mesh_h_p)
        I_xp = spline_I_x.ev(flat_mesh_w , flat_mesh_h)
        I_yp = spline_I_y.ev(flat_mesh_w , flat_mesh_h)
        
        
        
        
        
        #print(I_xp)
        
#        print(I_xp.shape)        
        G_I = []
        G_I.append(I_yp)
        G_I.append(I_xp)
        G_I = np.array(G_I)
        
        #G_I = np.concatenate((I_yp,I_xp))
        
        G_I_hello = np.stack((I_yp, I_xp), axis=-1)
        
        #print(G_I.shape)
        
        
        J = np.array([[1,0],[0,1]])

        H = (G_I_hello @ J).T @ (G_I_hello @ J)
        

        
#        IW = spline_It1.ev(mesh_w_p, mesh_h_p)
        IW = spline_It1.ev(flat_mesh_w, flat_mesh_h)
        
        dp = np.linalg.inv(H) @ ((G_I_hello @ J).T) @ (T - IW)
        
#        print(dp)
#        print(dp.shape)
#        
        p0[0] = p0[0] + dp[0]
        p0[1] = p0[1] + dp[1]
    
    return p0





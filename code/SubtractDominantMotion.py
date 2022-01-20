import numpy as np
from LucasKanadeAffine import LucasKanadeAffine

from scipy.ndimage.morphology import binary_dilation, binary_erosion
import scipy.ndimage
from InverseCompositionAffine import InverseCompositionAffine


def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)
    

    M = LucasKanadeAffine(image1, image2,threshold, num_iters)
    #Q3.1
    #M = InverseCompositionAffine(image1, image2,threshold, num_iters)
    
    
    
#    M_inverse = np.linalg.inv(M)
    h = image2.shape[0]
    w = image2.shape[1]
    image2_warp = scipy.ndimage.affine_transform(image2, M, output_shape=(h, w))
#    image2_warp = scipy.ndimage.affine_transform(image2,  M_inverse, output_shape=(h, w))
    image2_warp = binary_erosion(image2_warp)
    image2_warp = binary_dilation(image2_warp)

    diff = np.abs(image1 - image2_warp)

    
    mask = np.where(diff > tolerance,1,0)
    


    return mask

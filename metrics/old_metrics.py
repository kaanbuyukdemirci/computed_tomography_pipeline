import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
from skimage.metrics import structural_similarity as ssim

class Metrics(object):
    def __init__(self) -> None:
        pass
    
    def mse(self, image1, image2):
        return np.mean((image1 - image2)**2)

    def psnr(self, image1, image2):
        mse = self.mse(image1, image2)
        if mse == 0:
            return 100
        return min(20 * np.log10(255.0 / np.sqrt(mse)), 100)
    
    def ssim(self, image1, image2, data_range=1.0):
        return ssim(image1, image2, data_range=data_range)
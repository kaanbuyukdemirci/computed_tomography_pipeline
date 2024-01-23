import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from object_reconstructor import AbstractObjectReconstructor, AbstractReconstructionSettings

import numpy as np
from tqdm import tqdm
from skimage.transform import iradon

class SimulatorReconstructionSettings(AbstractReconstructionSettings):
    def __init__(self, original_shape: tuple) -> None:
        self.__original_shape = original_shape
        self.log = True
    
    @property
    def original_shape(self) -> tuple:
        return self.__original_shape

class SimulatorObjectReconstructor(AbstractObjectReconstructor):
    def __init__(self) -> None:
        pass
    
    def reconstruct_object(self, data: np.ndarray, angle_history: np.ndarray, 
                           reconstruction_settings:SimulatorReconstructionSettings) -> np.ndarray:
        # data : (z, y, x)
        # rotated around the z-axis
        # projected through y-axis
        # initial sinograms : (n_angles, z, x)
        
        # prepare
        sinograms = data.transpose(1, 2, 0) # (z, x, n_angles) where z is the rotation axis
        shape_difference = (np.array(data.shape[-1]) - np.array(reconstruction_settings.original_shape))[1:]
        pad_size = [int(i) for i in shape_difference/2]
        data = np.zeros(shape=reconstruction_settings.original_shape)
        if reconstruction_settings.log:
            progress_bar = tqdm(total=sinograms.shape[0], desc="Reconstructing 3D image", unit="sinogram", leave=True)
        
        # iterate over cross-sections
        for i, sinogram in enumerate(sinograms):
            image = iradon(sinogram, angle_history, circle=True, filter_name="ramp")
            data[i] = image[pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1]]
            if reconstruction_settings.log:
                progress_bar.update(1)
        
        # final adjustments
        # data = np.flip(data, axis=1) # thi might be necessary due to the implementation of iradon
        data = np.clip(data, 0, 1)
        
        return data 
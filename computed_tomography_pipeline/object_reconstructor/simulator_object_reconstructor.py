from .abstract_object_reconstructor import AbstractObjectReconstructor, AbstractReconstructionSettings

import numpy as np
from tqdm import tqdm
from skimage.transform import iradon

class SimulatorReconstructionSettings(AbstractReconstructionSettings):
    def __init__(self, original_shape: tuple, log: bool=True) -> None:
        self.__original_shape = original_shape
        self.log = log
    
    @property
    def original_shape(self) -> tuple:
        return self.__original_shape

class SimulatorObjectReconstructor(AbstractObjectReconstructor):
    def __init__(self) -> None:
        pass
    
    def reconstruct_object(self, data: np.ndarray, angle_history: np.ndarray, 
                           reconstruction_settings:SimulatorReconstructionSettings) -> np.ndarray:
        # data : (n, z, y, x)
        # rotated around the z-axis
        # projected through y-axis
        # initial sinograms : (n, n_angles, z, x)
        data = data.reshape(-1, *data.shape[-3:])
        
        # prepare
        sinograms = data.transpose(0, 2, 3, 1) # (n, z, x, n_angles) where z is the rotation axis
        shape_difference = (np.array(data.shape[-1]) - np.array(reconstruction_settings.original_shape))[-2:]
        pad_size = [int(i) for i in shape_difference/2]
        data = np.zeros(shape=(data.shape[0], *reconstruction_settings.original_shape))
        
        
        # iterate over cross-sections
        if False:
                sample_progress_bar = tqdm(total=sinograms.shape[0], desc="Reconstructing 3D image", unit="sample", leave=False)
        for object_i in range(data.shape[0]):
            if reconstruction_settings.log:
                progress_bar = tqdm(total=sinograms.shape[1], desc="Reconstructing 3D image", unit="cross-section", leave=False)
            for i, sinogram in enumerate(sinograms[object_i]):
                image = iradon(sinogram, angle_history, circle=True, filter_name="ramp")
                data[object_i, i] = image[pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1]]
                if reconstruction_settings.log:
                    progress_bar.update(1)
            if False:
                progress_bar.close()
                sample_progress_bar.update(1)
        
        # final adjustments
        # data = np.flip(data, axis=1) # this might be necessary due to the implementation of iradon
        data = np.clip(data, 0, 1)
        
        return data 
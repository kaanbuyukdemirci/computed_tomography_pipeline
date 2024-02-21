from .abstract_object_reconstructor import AbstractObjectReconstructor, AbstractReconstructionSettings

import numpy as np
from tqdm import tqdm
from skimage.transform import iradon
import multiprocessing

class SimulatorReconstructionSettings(AbstractReconstructionSettings):
    def __init__(self, log: bool=True, multiprocessing: bool=False) -> None:
        self.__original_shape = None
        self.log = log
        self.multiprocessing = multiprocessing
    
    @property
    def original_shape(self) -> tuple:
        return self.__original_shape
    
    def set_shape(self, shape: tuple) -> None:
        self.__original_shape = shape

def reconstruct_sinogram(sinogram, angle_history, pad_size):
    return iradon(sinogram, angle_history, circle=True, filter_name="ramp")[pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1]]
class SimulatorObjectReconstructor(AbstractObjectReconstructor):
    def __init__(self, reduce: int=1) -> None:
        self.reduce = reduce
    
    def reconstruct_object(self, initial_sinograms: np.ndarray, angle_history: np.ndarray, 
                           reconstruction_settings:SimulatorReconstructionSettings) -> np.ndarray:
        # data : (n, z, y, x)
        # rotated around the z-axis
        # projected through y-axis
        # initial sinograms : (n, n_angles, z, x)
        initial_sinograms = initial_sinograms.reshape(-1, *initial_sinograms.shape[-3:])
        
        # reduce the number of projections
        initial_sinograms = initial_sinograms[:, ::self.reduce]
        angle_history = angle_history[::self.reduce]

        # prepare
        sinograms = initial_sinograms.transpose(0, 2, 3, 1) # (n, z, x, n_angles) where z is the rotation axis
        shape_difference = (np.array(initial_sinograms.shape[-1]) - np.array(reconstruction_settings.original_shape))[-2:]
        pad_size = [int(i) for i in shape_difference/2]
        data = np.zeros(shape=(initial_sinograms.shape[0], *reconstruction_settings.original_shape))
        
        
        # iterate over cross-sections
        if reconstruction_settings.multiprocessing:
            with multiprocessing.Pool() as pool:
                for object_i in range(data.shape[0]):
                    if reconstruction_settings.log:
                        progress_bar = tqdm(total=sinograms.shape[1], desc="Reconstructing 3D image", unit="cross-section", leave=False)
                    for i, sinogram in enumerate(sinograms[object_i]):
                        image = pool.apply_async(reconstruct_sinogram, (sinogram, angle_history, pad_size))
                        data[object_i, i] = image.get()
                        if reconstruction_settings.log:
                            progress_bar.update(1)
        else:
            for object_i in range(data.shape[0]):
                if reconstruction_settings.log:
                    progress_bar = tqdm(total=sinograms.shape[1], desc="Reconstructing 3D image", unit="cross-section", leave=False)
                for i, sinogram in enumerate(sinograms[object_i]):
                    image = iradon(sinogram, angle_history, circle=True, filter_name="ramp")
                    data[object_i, i] = image[pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1]]
                    if reconstruction_settings.log:
                        progress_bar.update(1)
        
        # final adjustments
        # data = np.flip(data, axis=1) # this might be necessary due to the implementation of iradon
        data = np.clip(data, 0, 1)
        return data  
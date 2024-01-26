from .abstract_simulator import AbstractSimulator

import numpy as np
import torch
from torchvision.transforms.functional import rotate
import os
from typing import Callable
import shutil

class SimpleSimulator(AbstractSimulator):
    def __init__(self, data_generator_function:Callable, current_object_index:int=0, log:bool=True, cache:bool=True, clear_cache:bool=False) -> None:
        self.log = log
        self.cache = cache
        self.__clear_cache = clear_cache
        self.__cache_dir = "xray_cache/data/"
        self.__data_generator_function = data_generator_function
        self.current_object_index = current_object_index
        self.current_angle = 0
        self.__current_objects = None
        self.__object_shape = None
        self.__object_resolution = None
        self.__detector_resolution = None
        self.__xray_projection_shape = None
        self.initialize_dataset()
    
    @property
    def current_objects(self) -> np.ndarray:
        """The current objects to use.

        Returns
        -------
        np.ndarray, shape (n_objects, z, y, x)
            The current objects to use. The shape of the objects is (z, y, x).
        """
        return self.__current_objects
    
    @property
    def object_shape(self) -> tuple[int, int]:
        return self.__object_shape
    
    @property
    def object_resolution(self) -> int:
        return self.__object_resolution
    
    @property
    def detector_resolution(self) -> int:
        return self.__detector_resolution
    
    @property
    def xray_projection_shape(self) -> tuple[int, int, int]:
        return self.__xray_projection_shape
    
    def initialize_dataset(self):
        if self.__clear_cache:
            shutil.rmtree(self.__cache_dir)
        if self.cache:
            os.makedirs(self.__cache_dir, exist_ok=True)
        if os.path.exists(self.__cache_dir + "objects_0.npy"):
            pass
        else:
            if self.cache:
                os.makedirs(self.__cache_dir, exist_ok=True)
            # read the first image to get the shape and pixel range
            
            counter = 0
            while True:
                try:
                    np.save(self.__cache_dir + f"objects_{counter}.npy", next(self.__data_generator_function))
                    if self.log:
                        print(" " * 100, end="\r")
                        print(f"Reading image {counter}", end="\r")
                    counter += 1
                except StopIteration:
                    if self.log: print(" " * 100, end="\r")
                    break
        
        # set other properties
        self.__current_objects = np.load(self.__cache_dir + f"objects_{self.current_object_index}.npy")
        self.__object_shape = self.__current_objects.shape
        self.__object_resolution = max(self.__current_objects.shape[-2:]) # assumes square images
        res = np.ceil(self.object_resolution * np.sqrt(2)).astype(int)
        if (res % 2 == 1) and (self.object_resolution % 2 == 0): res += 1
        elif (res % 2 == 0) and (self.object_resolution % 2 == 1): res += 1
        self.__detector_resolution = res
        self.__xray_projection_shape = (len(self.__current_objects), self.detector_resolution, self.detector_resolution)
        
    def get_xray_projection(self) -> np.ndarray:
        """Get the x-ray projection of the current image at the given angle (self.current_angle).
        The motor rotates around the z-axis, and the x-ray is projected through the y-axis.
        
        Returns
        -------
        np.ndarray
            The x-ray projection of the current objects at given angles.
        """
        # prepare
        objects = torch.from_numpy(self.current_objects)
        angles = self.current_angle
        detector_resolution = self.detector_resolution
        resolution_difference = detector_resolution - self.object_resolution
        pad_size = int(resolution_difference / 2)
        rotated_objects = torch.zeros(objects.shape[0], detector_resolution, detector_resolution)
        rotated_objects[:, pad_size:-pad_size, pad_size:-pad_size] = objects
        del objects
        
        # rotate (- for clockwise, + for counter-clockwise)
        rotated_objects = rotate(rotated_objects, -angles)
        
        # project
        rotated_objects = rotated_objects.numpy()
        rotated_objects = rotated_objects.sum(axis=1)
        return rotated_objects
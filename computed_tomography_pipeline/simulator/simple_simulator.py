# supports return from object_generator_function of shape (..., z, y, x)

from .abstract_simulator import AbstractSimulator

import numpy as np
import torch
from torchvision.transforms.functional import rotate
from typing import Callable

class SimpleSimulator(AbstractSimulator):
    def __init__(self, object_generator_function:Callable, log:bool=True) -> None:
        self.__object_generator = object_generator_function
        self.log = log
        self.current_angle = 0
        self.__current_object = None
        self.__object_shape = None
        self.__xray_projection_shape = None
        self.initialize_new_object()
    
    @property
    def current_object(self) -> np.ndarray:
        return self.__current_object
    
    @property
    def object_shape(self) -> tuple[int, int]:
        return self.__object_shape
    
    @property
    def xray_projection_shape(self) -> int:
        return self.__xray_projection_shape
    
    def initialize_new_object(self) -> None:
        # get a sample
        self.__current_object = next(self.__object_generator)
        
        # set properties
        self.__object_shape = self.current_object.shape
        res = np.ceil(self.object_shape[-1] * np.sqrt(2)).astype(int)
        if (res % 2 == 1) and (self.object_shape[-1] % 2 == 0): res += 1
        elif (res % 2 == 0) and (self.object_shape[-1] % 2 == 1): res += 1
        self.__xray_projection_shape = (*self.current_object.shape[:-2], res) # assumes square detector
        
    def get_xray_projection(self) -> np.ndarray:
        """Get the x-ray projection of the current image at the given angle (self.current_angle).
        The motor rotates around the z-axis, and the x-ray is projected through the y-axis.
        """
        # prepare
        object = torch.from_numpy(self.current_object)
        angles = self.current_angle
        
        xray_projection_resolution = self.xray_projection_shape[-1]
        object_resolution = self.object_shape[-1]
        resolution_difference = xray_projection_resolution - object_resolution
        pad_size = int(resolution_difference / 2)
        
        rotated_object = torch.zeros(*object.shape[:-2], xray_projection_resolution, xray_projection_resolution)
        rotated_object[:, pad_size:-pad_size, pad_size:-pad_size] = object
        del object
        
        # rotate (- for clockwise, + for counter-clockwise)
        rotated_object = rotate(rotated_object, -angles)
        
        # project
        rotated_object = rotated_object.numpy()
        rotated_object = rotated_object.sum(axis=-2)
        
        return rotated_object
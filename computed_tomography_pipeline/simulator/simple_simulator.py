# supports return from object_generator_function of shape (..., z, y, x)

from .abstract_simulator import AbstractSimulator

import numpy as np
import torch
from torchvision.transforms.functional import rotate
from typing import Callable
import os
import pydicom as dicom

class SimpleDataset(object):
    def __init__(self, parent_path: str, cross_section_count: int) -> None:
        self.__parent_path = parent_path
        self.__cross_section_count = cross_section_count
        self.__init_dataset()
    
    def __init_dataset(self) -> None:
        path = []
        for path_i in os.listdir(self.__parent_path):
            path_i = self.__parent_path + path_i
            for path_j in os.listdir(path_i):
                path_j = path_i + "/" + path_j
                if len(os.listdir(path_j)) != self.__cross_section_count:
                    continue
                else:
                    path.append(path_j)
        self.__path = path
    
    def __data_reader(self, path):
        complete_data_paths = [os.path.join(path, data_path) 
                            for data_path in os.listdir(path) 
                            if data_path.endswith(".dcm")]
        sample = dicom.dcmread(complete_data_paths[0])
        shape = (len(complete_data_paths), sample.Rows, sample.Columns)
        pixel_range = [None, None]
        if sample.PixelRepresentation == 0:
            pixel_range[0] = 0
            pixel_range[1] = 2**sample.BitsStored - 1
        else:
            pixel_range[0] = -2**(sample.BitsStored - 1)
            pixel_range[1] = 2**(sample.BitsStored - 1) - 1
        
        data = np.zeros(shape)
        for i, path in enumerate(complete_data_paths):
            data[i] = (dicom.dcmread(path).pixel_array - pixel_range[0]) / (pixel_range[1] - pixel_range[0])
        
        return data
    
    def __getitem__(self, index:int) -> np.ndarray:
        return self.__data_reader(self.__path[index])

    def __len__(self) -> int:
        return len(self.__path)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class SimpleSimulator(AbstractSimulator):
    def __init__(self, iterable_generator_class:SimpleDataset, log:bool=True) -> None:
        self.__iterable_generator_class = iterable_generator_class
        self.__object_generator = self.__object_generator_function()
        self.log = log
        self.current_angle = 0
        self.__number_of_objects = len(self.__iterable_generator_class)
        self.__current_object = None
        self.__object_shape = None
        self.__xray_projection_shape = None
        self.initialize_new_object()
    
    def __object_generator_function (self):
        for i in self.__iterable_generator_class:
            yield i

    @property
    def number_of_objects(self) -> int:
        return self.__number_of_objects
    
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
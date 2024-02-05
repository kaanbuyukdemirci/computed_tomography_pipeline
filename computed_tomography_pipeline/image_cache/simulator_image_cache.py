from typing import Any
from .abstract_image_cache import AbstractImageCache
from ..xray_controller import SimulatorXrayController
from ..field_resolver import SimulatorFieldResolver
from ..angle_resolver import SimulatorAngleResolver

import numpy as np
import h5py
import copy

class MemoryManager(object):
    def __init__(self, memory_shape:np.ndarray) -> None:
        self.__memory_shape = memory_shape
        self.__occupancy_pointer = np.zeros_like(memory_shape)
        self.__is_memory_full = False

    def __call__(self, data_shape:np.ndarray) -> np.ndarray:
        if self.__is_memory_full:
            raise ValueError("Memory is full.")
        
        # 0 - adjust data shape
        data_shape = [i for i in data_shape if i != 1]
        if len(data_shape) == 0:
            data_shape = [1]
        
        # 1 - check for compatibility
        if not(all([True if data_shape[-i] <= self.__memory_shape[-i] else False for i in range(1, len(data_shape))]) \
            or len(data_shape) == 1):
                raise ValueError("Wrong shape.")
        if not(all([True if self.__occupancy_pointer[-i] == 0 else False for i in range(1, len(data_shape))]) \
            or len(data_shape) == 1):
                raise ValueError("Wrong shape.")
        if data_shape[0] > (self.__memory_shape[-len(data_shape)] - self.__occupancy_pointer[-len(data_shape)]):
            raise ValueError("Wrong shape.")
        
        # 2 - calculate range and determine start and end locations
        occupancy_pointer = self.__occupancy_pointer.copy()
        start_pointer = occupancy_pointer[:1-len(data_shape)] if len(data_shape) != 1 else occupancy_pointer
        end_pointer = start_pointer.copy()
        end_pointer[-1] += data_shape[0]
        self.__occupancy_pointer[:len(end_pointer)] = end_pointer
        for i in range(1, len(self.__occupancy_pointer) + 1):
            if self.__occupancy_pointer[-i] == self.__memory_shape[-i]:
                self.__occupancy_pointer[-i] = 0
                if i == len(self.__occupancy_pointer):
                    self.__is_memory_full = True
                else:
                    self.__occupancy_pointer[-i-1] += 1
        
        # 3 - create the slice
        slices = [slice(start_pointer[i], end_pointer[i], 1) if i == len(start_pointer)-1 else start_pointer[i]
                  for i in range(len(start_pointer))]
        return slices
class SimulatorImageCache(AbstractImageCache):
    def __init__(self, xray_controller:SimulatorXrayController, field_resolver:SimulatorFieldResolver,
                 angle_resolver:SimulatorAngleResolver, big_data_dictionary_path:str, number_of_objects: int=1, 
                 from_saved:bool=False) -> None:
        self.__xray_controller = xray_controller
        self.__field_resolver = field_resolver
        self.__angle_resolver = angle_resolver
        self.__big_data_dictionary_path = big_data_dictionary_path
        self.__from_saved = from_saved
        self.__number_of_objects = number_of_objects
        number_of_angles = self.__angle_resolver.number_of_angles
        self.__shapes = {"original_object": (self.__number_of_objects, *self.__xray_controller.identification.object_shape),
                         "flat_field_images": self.__field_resolver.flat_field_shape,
                         "dark_field_images": self.__field_resolver.dark_field_shape,
                         "projection_images": (self.__number_of_objects, number_of_angles, *self.__xray_controller.identification.xray_projection_shape),
                         "angles": (number_of_angles,),
                         "pre_processed_projection_images": (self.__number_of_objects, number_of_angles, *self.__xray_controller.identification.xray_projection_shape),
                         "reconstructed_object": (self.__number_of_objects, *self.__xray_controller.identification.object_shape)}
        self.__cache_pointers = {"original_object": MemoryManager(self.__shapes["original_object"]),
                           "flat_field_images": MemoryManager(self.__shapes["flat_field_images"]),
                           "dark_field_images": MemoryManager(self.__shapes["dark_field_images"]),
                           "projection_images": MemoryManager(self.__shapes["projection_images"]),
                           "angles": MemoryManager(self.__shapes["angles"]),
                           "pre_processed_projection_images": MemoryManager(self.__shapes["pre_processed_projection_images"]),
                           "reconstructed_object": MemoryManager(self.__shapes["reconstructed_object"])}
        self.__retrieval_pointers = copy.deepcopy(self.__cache_pointers)
        if not from_saved:
            self._AbstractImageCache__initialize_big_data_dictionary()
    
    def _AbstractImageCache__initialize_big_data_dictionary(self) -> None:
        with h5py.File(self.__big_data_dictionary_path, 'w') as f:
            f.create_dataset('original_object', shape=self.__shapes["original_object"], dtype=np.float32)
            f.create_dataset('flat_field_images', shape=self.__shapes["flat_field_images"], dtype=np.float32)
            f.create_dataset('dark_field_images', shape=self.__shapes["dark_field_images"], dtype=np.float32)
            f.create_dataset('projection_images', shape=self.__shapes["projection_images"], dtype=np.float32)
            f.create_dataset('angles', shape=self.__shapes["angles"], dtype=np.float32)
            f.create_dataset('pre_processed_projection_images', shape=self.__shapes["pre_processed_projection_images"], dtype=np.float32)
            f.create_dataset('reconstructed_object', shape=self.__shapes["reconstructed_object"], dtype=np.float32)

    @property
    def from_saved(self) -> bool:
        return self.__from_saved
    
    @property
    def shapes(self) -> dict:
        return self.__shapes
    
    @property
    def number_of_objects(self) -> int:
        return self.__number_of_objects
    
    def cache_data(self, name: str, data: np.ndarray) -> None:
        with h5py.File(self.__big_data_dictionary_path, 'r+') as f:
            dataset = f[name]
            slc = self.__cache_pointers[name](data.shape)
            dataset[*slc] = data
    
    def retrieve_data(self, name: str, shape: np.ndarray) -> np.ndarray:
        with h5py.File(self.__big_data_dictionary_path, 'r') as f:
            dataset = f[name]
            slc = self.__retrieval_pointers[name](shape)
            data = dataset[*slc]
        return data

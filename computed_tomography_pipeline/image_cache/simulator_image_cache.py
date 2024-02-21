from typing import Any
from .abstract_image_cache import AbstractImageCache
from ..xray_controller import SimulatorXrayController
from ..field_resolver import SimulatorFieldResolver
from ..angle_resolver import SimulatorAngleResolver

import numpy as np
import h5py
import copy
from typing import Optional

class MemoryManager(object):
    def __init__(self, memory_shape:np.ndarray) -> None:
        self.__memory_shape = memory_shape
        self.__occupancy_pointer = np.zeros_like(memory_shape)
        self.__is_memory_full = False
    
    @property
    def occupancy_pointer(self) -> np.ndarray:
        return self.__occupancy_pointer
    @property
    def is_memory_full(self) -> bool:
        return self.__is_memory_full
    @property
    def memory_shape(self) -> np.ndarray:
        return self.__memory_shape

    def set_occupancy_pointer(self, occupancy_pointer:np.ndarray) -> None:
        # 0 - adjust the occupancy_pointer shape
        occupancy_pointer = [occupancy_pointer[i] if i < len(occupancy_pointer) else 0 for i in range(len(self.__memory_shape))]
        self.__occupancy_pointer = np.array(occupancy_pointer)
        # TODO: check if memory is full
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
        #if len(data_shape) <= len(self.__occupancy_pointer):
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

    def __str__(self) -> str:
        return f"Memory Manager: memory_shape: {self.__memory_shape}, occupancy_pointer: {self.__occupancy_pointer}, is_memory_full: {self.__is_memory_full}"

class SimulatorImageCache(AbstractImageCache):
    def __init__(self, big_data_dictionary_path:str,
                 xray_controller:Optional[SimulatorXrayController], 
                 field_resolver:Optional[SimulatorFieldResolver],
                 angle_resolver:Optional[SimulatorAngleResolver], 
                 number_of_objects: Optional[int]=1, 
                 number_of_angle_resolvers: Optional[int]=1, 
                 number_of_pre_processors: Optional[int]=1,
                 number_of_reconstructors: Optional[int]=1, 
                 from_saved:bool=False) -> None:
        self.__big_data_dictionary_path = big_data_dictionary_path
        self.__from_saved = from_saved
        self.__number_of_objects_index = 0
        self.__number_of_angle_resolvers_index = 1
        self.__number_of_pre_processors_index = 2
        self.__number_of_reconstructors_index = 3
        if not from_saved:
            self.__xray_controller = xray_controller
            self.__field_resolver = field_resolver
            self.__angle_resolver = angle_resolver
            self.__number_of_objects = number_of_objects
            self.__number_of_angle_resolvers = number_of_angle_resolvers
            self.__number_of_pre_processors = number_of_pre_processors
            self.__number_of_reconstructors = number_of_reconstructors
        if self.__from_saved:
            # read
            with h5py.File(self.__big_data_dictionary_path, 'r') as f:
                self.__shapes = {"original_object": f['original_object'].shape,
                                 "flat_field_images": f['flat_field_images'].shape,
                                 "dark_field_images": f['dark_field_images'].shape,
                                 "angles": f['angles'].shape,
                                 "projection_images": f['projection_images'].shape,
                                 "pre_processed_projection_images": f['pre_processed_projection_images'].shape,
                                 "reconstructed_object": f['reconstructed_object'].shape,
                                 "original_object": f['original_object'].shape,
                                 "original_object": f['original_object'].shape,
                                 "original_object": f['original_object'].shape,
                                 "original_object": f['original_object'].shape,
                                 }
            self.__number_of_objects = self.__shapes["original_object"][0]
            self.__number_of_angle_resolvers = self.__shapes["angles"][1]
            self.__number_of_pre_processors = self.__shapes["pre_processed_projection_images"][2]
            self.__number_of_reconstructors = self.__shapes["reconstructed_object"][3]
            number_of_angles = self.__shapes["angles"][2]
        else:
            number_of_angles = [i.number_of_angles for i in self.__angle_resolver]
            if all([i==number_of_angles[0] for i in number_of_angles]):
                number_of_angles = number_of_angles[0]
            else:
                raise ValueError("Number of angles must be the same for all angle resolvers.")
            self.__shapes = {"original_object": (self.__number_of_objects, *self.__xray_controller.identification.object_shape),
                            "flat_field_images": (self.__number_of_objects, *self.__field_resolver.flat_field_shape),
                            "dark_field_images": (self.__number_of_objects, *self.__field_resolver.dark_field_shape),
                            "angles": (self.__number_of_objects, self.__number_of_angle_resolvers, number_of_angles,),
                            "projection_images": (self.__number_of_objects, self.__number_of_angle_resolvers, number_of_angles, *self.__xray_controller.identification.xray_projection_shape),
                            "pre_processed_projection_images": (self.__number_of_objects, self.__number_of_angle_resolvers, self.__number_of_pre_processors, number_of_angles, *self.__xray_controller.identification.xray_projection_shape),
                            "reconstructed_object": (self.__number_of_objects, self.__number_of_angle_resolvers, self.__number_of_pre_processors, self.__number_of_reconstructors, *self.__xray_controller.identification.object_shape)}

        self.cache_pointers = {"original_object": MemoryManager(self.__shapes["original_object"]),
                        "flat_field_images": MemoryManager(self.__shapes["flat_field_images"]),
                        "dark_field_images": MemoryManager(self.__shapes["dark_field_images"]),
                        "projection_images": MemoryManager(self.__shapes["projection_images"]),
                        "angles": MemoryManager(self.__shapes["angles"]),
                        "pre_processed_projection_images": MemoryManager(self.__shapes["pre_processed_projection_images"]),
                        "reconstructed_object": MemoryManager(self.__shapes["reconstructed_object"])}
        self.retrieval_pointers = copy.deepcopy(self.cache_pointers)
        if not self.__from_saved:
            self._AbstractImageCache__initialize_big_data_dictionary()
        self.__number_of_angles = number_of_angles

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
    @property
    def number_of_angle_resolvers(self) -> int:
        return self.__number_of_angle_resolvers
    @property
    def number_of_pre_processors(self) -> int:
        return self.__number_of_pre_processors
    @property
    def number_of_reconstructors(self) -> int:
        return self.__number_of_reconstructors
    @property
    def number_of_angles(self) -> int:
        return self.__number_of_angles

    def cache_data(self, name: str, data: np.ndarray) -> None:
        with h5py.File(self.__big_data_dictionary_path, 'r+') as f:
            dataset = f[name]
            slc = self.cache_pointers[name](data.shape)
            #print("cache", name, self.cache_pointers[name], slc, data.shape)
            dataset[*slc] = data
    def retrieve_data(self, name: str, shape: np.ndarray) -> np.ndarray:
        with h5py.File(self.__big_data_dictionary_path, 'r') as f:
            dataset = f[name]
            slc = self.retrieval_pointers[name](shape)
            #print("retrieve", name, self.retrieval_pointers[name], slc)
            data = dataset[*slc]
        return data

    def set_all_occupancy_pointers(self, occupancy_pointer: np.ndarray) -> None:
        for i in self.cache_pointers.keys():
            occupancy_pointer_size = self.cache_pointers[i].occupancy_pointer.size
            new_occupancy_pointer = [occupancy_pointer[j] if j < len(occupancy_pointer) else 0 for j in range(occupancy_pointer_size)]
            self.cache_pointers[i].set_occupancy_pointer(new_occupancy_pointer)
            self.retrieval_pointers[i].set_occupancy_pointer(new_occupancy_pointer)
    def reset_all_occupancy_pointers(self) -> None:
        self.set_all_occupancy_pointers(np.array([]))
    def expand_memory(self, name: str, shape: np.ndarray) -> None:
        # create new dataset
        with h5py.File(self.__big_data_dictionary_path, 'r+') as f:
            f.create_dataset(name+"_new", shape=shape, dtype=np.float32)
        # copy data
        with h5py.File(self.__big_data_dictionary_path, 'r+') as f:
            dataset = f[name]
            new_dataset = f[name+"_new"]
            slices = [0 if i < len(shape)-len(dataset.shape) else slice(0, dataset.shape[i], 1) for i in range(len(shape))]
            first_slice_array = np.arange(slices[0].start, slices[0].stop, slices[0].step)
            for i in first_slice_array:
                slices[0] = i
                new_dataset[*slices] = dataset[i]
            # delete old dataset
            del f[name]
            # rename new dataset
            f.move(name+"_new", name)
        # update shapes
        self.__shapes[name] = shape
        self.cache_pointers[name] = MemoryManager(self.__shapes[name])
        self.retrieval_pointers[name] = MemoryManager(self.__shapes[name])
        # set pointer to the start of the expanded memory
        # self.set_all_occupancy_pointers

    def set_pointers_for_ith_object(self, i: int) -> None:
        keys = ["original_object", "flat_field_images", "dark_field_images", "projection_images", "angles", "pre_processed_projection_images", "reconstructed_object"]
        for key in keys:
            cache_pointer = self.cache_pointers[key].occupancy_pointer
            cache_pointer[self.__number_of_objects_index] = i
            cache_pointer = cache_pointer[:self.__number_of_objects_index+1]
            retrieval_pointer = self.retrieval_pointers[key].occupancy_pointer
            retrieval_pointer[self.__number_of_objects_index] = i
            retrieval_pointer = retrieval_pointer[:self.__number_of_objects_index+1]
            self.cache_pointers[key].set_occupancy_pointer(cache_pointer)
            self.retrieval_pointers[key].set_occupancy_pointer(retrieval_pointer)
    def set_pointers_for_ith_angle_resolver(self, i: int) -> None:
        keys = ["angles", "projection_images", "pre_processed_projection_images", "reconstructed_object"]
        for key in keys:
            cache_pointer = self.cache_pointers[key].occupancy_pointer
            cache_pointer[self.__number_of_angle_resolvers_index] = i
            cache_pointer = cache_pointer[:self.__number_of_angle_resolvers_index+1]
            retrieval_pointer = self.retrieval_pointers[key].occupancy_pointer
            retrieval_pointer[self.__number_of_angle_resolvers_index] = i
            retrieval_pointer = retrieval_pointer[:self.__number_of_angle_resolvers_index+1]
            self.cache_pointers[key].set_occupancy_pointer(cache_pointer)
            self.retrieval_pointers[key].set_occupancy_pointer(retrieval_pointer)
    def set_pointers_for_ith_pre_processor(self, i: int) -> None:
        keys = ["pre_processed_projection_images", "reconstructed_object"]
        for key in keys:
            cache_pointer = self.cache_pointers[key].occupancy_pointer
            cache_pointer[self.__number_of_pre_processors_index] = i
            cache_pointer = cache_pointer[:self.__number_of_pre_processors_index+1]
            retrieval_pointer = self.retrieval_pointers[key].occupancy_pointer
            retrieval_pointer[self.__number_of_pre_processors_index] = i
            retrieval_pointer = retrieval_pointer[:self.__number_of_pre_processors_index+1]
            self.cache_pointers[key].set_occupancy_pointer(cache_pointer)
            self.retrieval_pointers[key].set_occupancy_pointer(retrieval_pointer)
    def set_pointers_for_ith_reconstructor(self, i: int) -> None:
        keys = ["reconstructed_object"]
        for key in keys:
            cache_pointer = self.cache_pointers[key].occupancy_pointer
            cache_pointer[self.__number_of_reconstructors_index] = i
            cache_pointer = cache_pointer[:self.__number_of_reconstructors_index+1]
            retrieval_pointer = self.retrieval_pointers[key].occupancy_pointer
            retrieval_pointer[self.__number_of_reconstructors_index] = i
            retrieval_pointer = retrieval_pointer[:self.__number_of_reconstructors_index+1]
            self.cache_pointers[key].set_occupancy_pointer(cache_pointer)
            self.retrieval_pointers[key].set_occupancy_pointer(retrieval_pointer)

    def add_angle_resolver(self, new_number_of_angle_resolvers: int) -> None:
        keys = ["angles", "projection_images", "pre_processed_projection_images", "reconstructed_object"]
        for key in keys:
            # read the current shape
            with h5py.File(self.__big_data_dictionary_path, 'r') as f:
                shape = f[key].shape
            # expand the shape
            shape = list(shape)
            shape[self.__number_of_angle_resolvers_index] = new_number_of_angle_resolvers
            shape = tuple(shape)
            # expand memory
            self.expand_memory(key, shape)
        #self.set_pointers_for_ith_angle_resolver(self.__number_of_angle_resolvers)
        self.__number_of_angle_resolvers = new_number_of_angle_resolvers
    def add_pre_processor(self, new_number_of_pre_processors: int) -> None:
        keys = ["pre_processed_projection_images", "reconstructed_object"]
        for key in keys:
            # read the current shape
            with h5py.File(self.__big_data_dictionary_path, 'r') as f:
                shape = f[key].shape
            # expand the shape
            shape = list(shape)
            shape[self.__number_of_pre_processors_index] = new_number_of_pre_processors
            shape = tuple(shape)
            # expand memory
            self.expand_memory(key, shape)
        #self.set_pointers_for_ith_pre_processor(self.__number_of_pre_processors)
        self.__number_of_pre_processors = new_number_of_pre_processors
    def add_reconstructor(self, new_number_of_reconstructors: int) -> None:
        keys = ["reconstructed_object"]
        for key in keys:
            # read the current shape
            with h5py.File(self.__big_data_dictionary_path, 'r') as f:
                shape = f[key].shape
            # expand the shape
            shape = list(shape)
            shape[self.__number_of_reconstructors_index] = new_number_of_reconstructors
            shape = tuple(shape)
            # expand memory
            self.expand_memory(key, shape)
        #self.set_pointers_for_ith_pre_processor(self.__number_of_reconstructors)
        self.__number_of_reconstructors = new_number_of_reconstructors

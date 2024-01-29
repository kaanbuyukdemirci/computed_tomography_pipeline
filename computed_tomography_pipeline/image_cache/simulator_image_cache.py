from .abstract_image_cache import AbstractImageCache

import numpy as np

class SimulatorImageCache(AbstractImageCache):
    def __init__(self) -> None:
        self.__projection_images = []
        self.__dark_images = None
        self.__light_images = None
    
    @property
    def projection_images(self) -> np.ndarray:
        return self.__projection_images
    
    def cache_field_images(self, dark_image: np.ndarray, light_image: np.ndarray) -> None:
        self.__dark_images = dark_image
        self.__light_images = light_image
    
    def cache_projection_images(self, image: np.ndarray) -> None:
        self.__projection_images.append(image)
    
    def finalize_projection_images(self) -> None:
        self.__projection_images = np.array(self.__projection_images)
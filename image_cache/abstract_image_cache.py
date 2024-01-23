import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from abc import ABC, abstractmethod
import numpy as np

class AbstractImageCache(ABC):
    def __init__(self) -> None:
        self.__projection_images = []
        self.__dark_images = None
        self.__light_images = None
    
    @property
    @abstractmethod
    def projection_images(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def cache_field_images(self, dark_image: np.ndarray, light_image: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def cache_projection_images(self, image: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def finalize_projection_images(self) -> None:
        pass
    
from abc import ABC, abstractmethod
import numpy as np
from typing import Callable

class AbstractSimulator(ABC):
    object_generator_function:Callable
    
    @property
    @abstractmethod
    def current_object(self) -> np.ndarray:
        pass
    
    @property
    @abstractmethod
    def object_shape(self) -> tuple[int, int, int]:
        pass
    
    @property
    @abstractmethod
    def xray_projection_shape(self) -> tuple[int, int, int]:
        pass
    
    @abstractmethod
    def get_xray_projection(self) -> np.ndarray:
        pass
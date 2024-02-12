from abc import ABC, abstractmethod
import numpy as np
from typing import Iterable

class AbstractSimulator(ABC):
    iterable_generator_class: Iterable
    
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
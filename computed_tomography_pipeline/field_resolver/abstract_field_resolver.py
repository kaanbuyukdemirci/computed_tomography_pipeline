from ..xray_controller import AbstractXrayController

from abc import ABC, abstractmethod
import numpy as np
from typing import Iterable

class AbstractFieldResolver(ABC):
    __xray_controller: AbstractXrayController
    dark_field_shape: Iterable[int]
    flat_field_shape: Iterable[int]
    
    @abstractmethod
    def get_field_images(self) -> tuple[np.ndarray, np.ndarray]:
        pass
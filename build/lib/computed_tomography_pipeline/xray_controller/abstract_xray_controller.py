from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class AbstractXraySetting(ABC):
    pass

class AbstractXrayController(ABC):
    __identification: Any
    
    @abstractmethod
    def get_image(self, xray_setting:AbstractXraySetting) -> np.ndarray:
        pass
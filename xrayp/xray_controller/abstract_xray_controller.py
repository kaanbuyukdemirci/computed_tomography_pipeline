from abc import ABC, abstractmethod
import numpy as np

class AbstractXraySetting(ABC):
    pass

class AbstractXrayController(ABC):
    def __init__(self, identification) -> None:
        super().__init__()
        self.__identification = identification
    
    @abstractmethod
    def get_image(self, xray_setting:AbstractXraySetting) -> np.ndarray:
        pass
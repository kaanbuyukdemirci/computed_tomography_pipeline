from abc import ABC, abstractmethod
import numpy as np

class AbstractReconstructionSettings(ABC):
    pass

class AbstractObjectReconstructor(ABC):
    def __init__(self, settings: AbstractReconstructionSettings) -> None:
        self.__settings = settings
    
    @abstractmethod
    def reconstruct_object(self, data: np.ndarray, reconstruction_settings: AbstractReconstructionSettings) -> np.ndarray:
        pass
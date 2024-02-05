from ..xray_controller import AbstractXrayController
from ..field_resolver import AbstractFieldResolver
from ..angle_resolver import AbstractAngleResolver

from abc import ABC, abstractmethod
import numpy as np

class AbstractImageCache(ABC):
    __xray_controller: AbstractXrayController
    __field_resolver: AbstractFieldResolver
    __angle_resolver: AbstractAngleResolver
    __big_data_dictionary_path: str
    
    @abstractmethod
    def __initialize_big_data_dictionary(self) -> None:
        pass
    
    @abstractmethod
    def cache_data(self, name: str, data: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def retrieve_data(self, name: str, index: np.ndarray) -> np.ndarray:
        pass
    
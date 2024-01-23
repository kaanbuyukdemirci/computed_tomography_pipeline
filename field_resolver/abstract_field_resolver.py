import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from xray_controller import AbstractXrayController

from abc import ABC, abstractmethod
import numpy as np

class AbstractFieldResolver(ABC):
    def __init__(self, xray_controller:AbstractXrayController):
        self.__xray_controller = xray_controller
    
    @abstractmethod
    def get_field_images(self) -> tuple[np.ndarray, np.ndarray]:
        pass
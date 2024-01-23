import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from abc import ABC, abstractmethod
import numpy as np

class AbstractImagePreprocessor(ABC):
    def __init__(self, preprocessor_function) -> None:
        self.__preprocessor_function = preprocessor_function

    @abstractmethod
    def preprocess_image(self, image) -> np.ndarray:
        pass
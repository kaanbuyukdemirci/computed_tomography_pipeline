from abc import ABC, abstractmethod
import numpy as np
from typing import Callable

class AbstractImagePreprocessor(ABC):
    __preprocessor_function: Callable[[np.ndarray], np.ndarray]

    @abstractmethod
    def preprocess_image(self, image) -> np.ndarray:
        pass
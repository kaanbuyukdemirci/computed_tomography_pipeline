from .abstract_image_preprocessor import AbstractImagePreprocessor

import numpy as np

class SimulatorImagePreprocessor(AbstractImagePreprocessor):
    def __init__(self, preprocessor_function=None) -> None:
        self.__preprocessor_function = lambda image : image

    def preprocess_image(self, image) -> np.ndarray:
        return self.__preprocessor_function(image)
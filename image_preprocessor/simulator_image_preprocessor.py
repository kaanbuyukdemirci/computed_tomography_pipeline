import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from image_preprocessor import AbstractImagePreprocessor

import numpy as np

class SimulatorImagePreprocessor(AbstractImagePreprocessor):
    def __init__(self, preprocessor_function=None) -> None:
        self.__preprocessor_function = lambda image : image

    def preprocess_image(self, image) -> np.ndarray:
        return self.__preprocessor_function(image)
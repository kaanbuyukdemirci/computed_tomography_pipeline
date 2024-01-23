import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from abc import ABC, abstractmethod
import numpy as np

from simulator import AbstractSimulator
from motor_controller import AbstractMotorController
from xray_controller import AbstractXrayController, AbstractXraySetting
from field_resolver import AbstractFieldResolver
from angle_resolver import AbstractAngleResolver
from image_cache import AbstractImageCache
from image_preprocessor import AbstractImagePreprocessor
from object_reconstructor import AbstractObjectReconstructor, AbstractReconstructionSettings

class AbstractPipelineSettings(ABC):
    pass

class AbstractXrayPipeline(ABC):
    def __init__(self, simulator: AbstractSimulator, motor_controller: AbstractMotorController, 
                 xray_controller: AbstractXrayController, field_resolver: AbstractFieldResolver, 
                 angle_resolver: AbstractAngleResolver, image_cache: AbstractImageCache, 
                 image_preprocessor: AbstractImagePreprocessor, object_reconstructor: AbstractObjectReconstructor) -> None:
        super().__init__()
        self.__simulator = simulator
        self.__motor_controller = motor_controller
        self.__xray_controller = xray_controller
        self.__field_resolver = field_resolver
        self.__angle_resolver = angle_resolver
        self.__image_cache = image_cache
        self.__image_preprocessor = image_preprocessor
        self.__object_reconstructor = object_reconstructor
    
    @abstractmethod
    def execute_pipeline(self, settings: AbstractPipelineSettings) -> np.ndarray:
        pass
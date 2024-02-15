from ..simulator import AbstractSimulator
from ..motor_controller import AbstractMotorController
from ..xray_controller import AbstractXrayController, AbstractXraySetting
from ..field_resolver import AbstractFieldResolver
from ..angle_resolver import AbstractAngleResolver
from ..image_cache import AbstractImageCache
from ..image_preprocessor import AbstractImagePreprocessor
from ..object_reconstructor import AbstractObjectReconstructor, AbstractReconstructionSettings

from abc import ABC, abstractmethod
import numpy as np

class AbstractPipelineSettings(ABC):
    pass

class AbstractCtPipeline(ABC):
    __simulator: AbstractSimulator
    __motor_controller: AbstractMotorController
    __xray_controller: AbstractXrayController
    __field_resolver: AbstractFieldResolver
    __angle_resolver: AbstractAngleResolver
    __image_cache: AbstractImageCache
    __image_preprocessor: AbstractImagePreprocessor
    __object_reconstructor: AbstractObjectReconstructor
    
    @abstractmethod
    def execute_pipeline(self, settings: AbstractPipelineSettings) -> np.ndarray:
        pass
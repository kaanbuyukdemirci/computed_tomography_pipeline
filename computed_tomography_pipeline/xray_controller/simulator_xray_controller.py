from .abstract_xray_controller import AbstractXrayController, AbstractXraySetting
from ..simulator import SimpleSimulator

import numpy as np
from typing import Literal

class SimulatorXraySetting(AbstractXraySetting):
    def __init__(self, power:Literal['on', 'off']) -> None:
        self.power = power

class SimulatorXrayController(AbstractXrayController):
    def __init__(self, identification:SimpleSimulator) -> None:
        self.__identification = identification
    
    @property
    def identification(self) -> SimpleSimulator:
        return self.__identification
    
    def get_image(self, xray_setting:SimulatorXraySetting) -> np.ndarray:
        if xray_setting.power == 'on':
            return self.__identification.get_xray_projection()
        else:
            return np.zeros(shape=self.__identification.xray_projection_shape)
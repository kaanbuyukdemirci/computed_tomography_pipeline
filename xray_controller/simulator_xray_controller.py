import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from xray_controller import AbstractXrayController, AbstractXraySetting
from simulator import SimpleSimulator

import numpy as np
from typing import Literal

class SimulatorXraySetting(AbstractXraySetting):
    def __init__(self, power:Literal['on', 'off']) -> None:
        self.__power= power
    
    @property
    def power(self) -> Literal['on', 'off']:
        return self.__power
    @power.setter
    def power(self, power:Literal['on', 'off']) -> None:
        self.__power = power

class SimulatorXrayController(AbstractXrayController):
    def __init__(self, identification:SimpleSimulator) -> None:
        self.__identification = identification
    
    def get_image(self, xray_setting:SimulatorXraySetting) -> np.ndarray:
        if xray_setting.power == 'on':
            return self.__identification.get_xray_projection()
        else:
            return np.zeros(shape=self.__identification.xray_projection_shape)
import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from abc import ABC, abstractmethod

class AbstractMotorController(ABC):
    def __init__(self, identification) -> None:
        self.__identification = identification
    
    @abstractmethod
    def rotate_motor(self, angle: float) -> None:
        pass
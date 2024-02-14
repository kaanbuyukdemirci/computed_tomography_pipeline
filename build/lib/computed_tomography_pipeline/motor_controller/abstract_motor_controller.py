from abc import ABC, abstractmethod
from typing import Any

class AbstractMotorController(ABC):
    __identification:Any
    
    @abstractmethod
    def rotate_motor(self, angle: float) -> None:
        pass
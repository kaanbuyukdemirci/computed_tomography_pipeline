import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from abc import ABC, abstractmethod

class AbstractAngleResolver(ABC):
    def __init__(self) -> None:
        self.__angle_history = list()
        self.__current_angle = None
    
    @property
    @abstractmethod
    def angle_history(self) -> list[float]:
        pass
    
    @property
    @abstractmethod
    def current_angle(self) -> float:
        pass
    
    @current_angle.setter
    @abstractmethod
    def current_angle(self, value):
        pass
    
    def resolve_angle(self, information) -> tuple[bool, float]:
        pass
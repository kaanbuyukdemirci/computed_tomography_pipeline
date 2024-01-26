from abc import ABC, abstractmethod

class AbstractMotorController(ABC):
    def __init__(self, identification) -> None:
        self.__identification = identification
    
    @abstractmethod
    def rotate_motor(self, angle: float) -> None:
        pass
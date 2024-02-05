from .abstract_angle_resolver import AbstractAngleResolver
from typing import Optional

class SimulatorAngleResolver(AbstractAngleResolver):
    def __init__(self, number_of_angles: int) -> None:
        self.__number_of_angles = number_of_angles
        self.__angle_history = list()
        self.__current_angle = None
        self.__angle_change = 180 / (self.__number_of_angles)
    
    @property
    def number_of_angles(self) -> int:
        return self.__number_of_angles
    @property
    def angle_history(self) -> list[float]:
        return self.__angle_history
        
    @property
    def current_angle(self) -> float:
        return self.__current_angle
    @current_angle.setter
    def current_angle(self, value):
        self.__angle_history.append(self.__current_angle)
        self.__current_angle = value
    
    def resolve_angle(self, information=None) -> Optional[float]:
        if self.current_angle is None:
            self.__current_angle = 0
        else:
            self.current_angle = self.current_angle + self.__angle_change
            if self.current_angle >= 180:
                return None
        return self.current_angle
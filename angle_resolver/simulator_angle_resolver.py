import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from angle_resolver import AbstractAngleResolver

class SimulatorAngleResolver(AbstractAngleResolver):
    def __init__(self, delta_theta: float) -> None:
        self.__delta_theta = delta_theta
        self.__angle_history = list()
        self.__current_angle = None
    
    @property
    def angle_history(self) -> list[float]:
        return self.__angle_history
        
    @property
    def current_angle(self) -> float:
        return self.__current_angle
    @current_angle.setter
    def current_angle(self, value):
        self.__current_angle = value
        self.__angle_history.append(value)
    
    def is_valid_angle_change(self) -> bool:
        return self.current_angle < 180
    
    def resolve_angle(self, information=None) -> tuple[bool, float]:
        if self.current_angle is None:
            self.current_angle = 0
        else:
            self.current_angle = self.current_angle + self.__delta_theta
        return (self.is_valid_angle_change(), self.current_angle)
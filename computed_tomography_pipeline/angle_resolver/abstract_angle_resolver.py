from abc import ABC, abstractmethod
from typing import Any, Optional

class AbstractAngleResolver(ABC):
    __angle_history: list[float] = list()
    __current_angle: Optional[float] = None
    __number_of_angles: Optional[int] = None
    
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
    
    def resolve_angle(self, information: Any=None) -> Optional[float]:
        pass
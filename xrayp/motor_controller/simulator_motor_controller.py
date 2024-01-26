from .abstract_motor_controller import AbstractMotorController
from ..simulator import SimpleSimulator

class SimulatorMotorController(AbstractMotorController):
    def __init__(self, identification: SimpleSimulator) -> None:
        self.__identification = identification
    
    def rotate_motor(self, angle: float) -> None:
        self.__identification.current_angle = angle
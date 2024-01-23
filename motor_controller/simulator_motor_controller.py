import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from simulator import SimpleSimulator
from motor_controller import AbstractMotorController

class SimulatorMotorController(AbstractMotorController):
    def __init__(self, identification: SimpleSimulator) -> None:
        self.__identification = identification
    
    def rotate_motor(self, angle: float) -> None:
        self.__identification.current_angles = angle
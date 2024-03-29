from .abstract_field_resolver import AbstractFieldResolver
from ..xray_controller import SimulatorXrayController, SimulatorXraySetting

class SimulatorFieldResolver(AbstractFieldResolver):
    def __init__(self, xray_controller:SimulatorXrayController):
        self.__xray_controller = xray_controller
        self.__dark_field_shape = self.__xray_controller.identification.xray_projection_shape
        self.__flat_field_shape = self.__dark_field_shape
    
    @property
    def dark_field_shape(self):
        return self.__dark_field_shape
    @property
    def flat_field_shape(self):
        return self.__flat_field_shape
    
    def get_field_images(self):
        # get 1 dark image (dark field)
        dark_setting = SimulatorXraySetting('off')
        dark_image = self.__xray_controller.get_image(dark_setting)
        
        # get 1 light image (flat field)
        light_setting = SimulatorXraySetting('on')
        light_image = self.__xray_controller.get_image(light_setting)
        
        return dark_image, light_image

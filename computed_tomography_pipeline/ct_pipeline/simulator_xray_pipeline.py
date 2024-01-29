from .abstract_xray_pipeline import AbstractCtPipeline, AbstractPipelineSettings
from ..simulator import SimpleSimulator
from ..motor_controller import SimulatorMotorController
from ..xray_controller import SimulatorXrayController, SimulatorXraySetting
from ..field_resolver import SimulatorFieldResolver
from ..angle_resolver import SimulatorAngleResolver
from ..image_cache import SimulatorImageCache
from ..image_preprocessor import SimulatorImagePreprocessor
from ..object_reconstructor import SimulatorObjectReconstructor, SimulatorReconstructionSettings

import numpy as np
from tqdm import tqdm
import os

class SimulatorPipelineSettings(AbstractPipelineSettings):
    def __init__(self) -> None:
        self.cache = True
        self.log = True

class SimulatorCtPipeline(AbstractCtPipeline):
    def __init__(self, simulator: SimpleSimulator, motor_controller: SimulatorMotorController, 
                 xray_controller: SimulatorXrayController, xray_setting: SimulatorXraySetting,
                 field_resolver: SimulatorFieldResolver, angle_resolver: SimulatorAngleResolver, 
                 image_cache: SimulatorImageCache, image_preprocessor: SimulatorImagePreprocessor, 
                 object_reconstructor: SimulatorObjectReconstructor, 
                 reconstruction_settings:SimulatorReconstructionSettings) -> None:
        self.__simulator = simulator
        self.__motor_controller = motor_controller
        self.__xray_controller = xray_controller
        self.__xray_setting = xray_setting
        self.__field_resolver = field_resolver
        self.__angle_resolver = angle_resolver
        self.__image_cache = image_cache
        self.__image_preprocessor = image_preprocessor
        self.__object_reconstructor = object_reconstructor
        self.__reconstruction_settings = reconstruction_settings
        self.__cache_dir = "ctp_cache/pipeline/"
        os.makedirs(self.__cache_dir, exist_ok=True)
    
    def execute_pipeline(self, settings: SimulatorPipelineSettings) -> np.ndarray:
        dark_image, light_image = self.__field_resolver.get_field_images()
        self.__image_cache.cache_field_images(dark_image, light_image)
        is_enough, angle = self.__angle_resolver.resolve_angle()
        if settings.log:
            progress_bar = tqdm(total=180, desc="Getting Xray Images", unit="scanned angle", leave=True)
        while is_enough:
            progress_bar.update(angle - progress_bar.n)
            self.__motor_controller.rotate_motor(angle)
            image = self.__xray_controller.get_image(self.__xray_setting)
            image = self.__image_preprocessor.preprocess_image(image)
            self.__image_cache.cache_projection_images(image)
            is_enough, angle = self.__angle_resolver.resolve_angle()
        if settings.log:
            progress_bar.update(progress_bar.total - progress_bar.n)
        self.__image_cache.finalize_projection_images()
        reconstruction = self.__object_reconstructor.reconstruct_object(data=self.__image_cache.projection_images,
                                                                        angle_history=self.__angle_resolver.angle_history[:-1],
                                                                        reconstruction_settings=self.__reconstruction_settings)
        if settings.cache:
            np.save(self.__cache_dir + "original_object.npy", self.__simulator.current_objects)
            np.save(self.__cache_dir + "xray_images.npy", self.__image_cache.projection_images)
            np.save(self.__cache_dir + "angles.npy", np.array(self.__angle_resolver.angle_history[:-1]))
            np.save(self.__cache_dir + "reconstruction.npy", reconstruction)
        return reconstruction

    
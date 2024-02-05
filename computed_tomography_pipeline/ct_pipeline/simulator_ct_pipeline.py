from .abstract_ct_pipeline import AbstractCtPipeline, AbstractPipelineSettings
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
from typing import Literal

class SimulatorPipelineSettings(AbstractPipelineSettings):
    def __init__(self, log: bool=True, start_from: Literal['beginning', 'projections', 'pre_processing']='beginning') -> None:
        self.log = log
        self.start_from = start_from

class SimulatorCtPipeline(AbstractCtPipeline):
    def __init__(self, simulator: SimpleSimulator, motor_controller: SimulatorMotorController, 
                 xray_controller: SimulatorXrayController, xray_setting: SimulatorXraySetting,
                 field_resolver: SimulatorFieldResolver, angle_resolver: SimulatorAngleResolver, 
                 image_cache: SimulatorImageCache, image_preprocessor: SimulatorImagePreprocessor, 
                 object_reconstructor: SimulatorObjectReconstructor) -> None:
        self.__simulator = simulator
        self.__motor_controller = motor_controller
        self.__xray_controller = xray_controller
        self.__xray_setting = xray_setting
        self.__field_resolver = field_resolver
        self.__angle_resolver = angle_resolver
        self.__image_cache = image_cache
        self.__image_preprocessor = image_preprocessor
        self.__object_reconstructor = object_reconstructor
        self.__reconstruction_settings = SimulatorReconstructionSettings(self.__simulator.object_shape, 
                                                                         log=True)
    
    def execute_pipeline(self, pipeline_settings: SimulatorPipelineSettings) -> np.ndarray:
        if (pipeline_settings.start_from != 'beginning') and not(self.__image_cache.from_saved):
            raise ValueError("Cannot start from a point in the pipeline that has not been cached.\n"+
                             "Please set from_saved to True in SimulatorImageCache,\n"+
                             "and make sure that the big_data_dictionary isn't accidentally emptied\n"+
                             "it is emptied if big_data_dictionary_path is the same as the path used in this run.")
        self.__reconstruction_settings.log = pipeline_settings.log
        first_run = True
        for i in range(self.__image_cache.number_of_objects):
            if pipeline_settings.start_from == 'beginning':
                self.__image_cache.cache_data("original_object", self.__simulator.current_object)
                if first_run:
                    flat_field, dark_field = self.__field_resolver.get_field_images()
                    self.__image_cache.cache_data("flat_field_images", flat_field)
                    self.__image_cache.cache_data("dark_field_images", dark_field)
                if first_run:
                    angle = self.__angle_resolver.resolve_angle()
                else:
                    angle_index = 0
                    angle = angles[angle_index]
                if pipeline_settings.log:
                    progress_bar = tqdm(total=180, desc="Getting Xray Images", unit="scanned angle", leave=False)
                while angle is not None:
                    if pipeline_settings.log:
                        progress_bar.update(angle - progress_bar.n)
                    self.__motor_controller.rotate_motor(angle)
                    image = self.__xray_controller.get_image(self.__xray_setting)
                    self.__image_cache.cache_data("projection_images", image)
                    if first_run:
                        angle = self.__angle_resolver.resolve_angle()
                    else:
                        angle_index += 1
                        angle = angles[angle_index] if angle_index < len(angles) else None
                if pipeline_settings.log:
                    progress_bar.update(progress_bar.total - progress_bar.n)
                    progress_bar.close()
                if first_run:
                    angles = np.array(self.__angle_resolver.angle_history)
                    self.__image_cache.cache_data("angles", np.array(self.__angle_resolver.angle_history))
            if (pipeline_settings.start_from == 'projections') or (pipeline_settings.start_from == 'beginning'):
                # divide the data into smaller chunks to avoid memory error)
                projection_shape = self.__image_cache.shapes["projection_images"][1:]
                data = self.__image_cache.retrieve_data("projection_images", projection_shape)
                data = self.__image_preprocessor.preprocess_image(data)
                self.__image_cache.cache_data("pre_processed_projection_images", data)
            if (pipeline_settings.start_from == 'pre_processing') or (pipeline_settings.start_from == 'projections') or (pipeline_settings.start_from == 'beginning'):
                projection_shape = self.__image_cache.shapes["projection_images"][1:]
                if first_run:
                    angle = self.__image_cache.retrieve_data("angles", [len(self.__angle_resolver.angle_history)])
                else: 
                    angle = angles
                data = self.__image_cache.retrieve_data("pre_processed_projection_images", projection_shape)
                reconstruction = self.__object_reconstructor.reconstruct_object(data=data, angle_history=angle, 
                                                                                reconstruction_settings=self.__reconstruction_settings)
                self.__image_cache.cache_data("reconstructed_object", reconstruction)
            if i != self.__image_cache.number_of_objects - 1:
                self.__simulator.initialize_new_object()
            first_run = False

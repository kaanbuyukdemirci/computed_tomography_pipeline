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
from typing import Literal, Optional

class SimulatorPipelineSettings(AbstractPipelineSettings):
    def __init__(self, log: bool=True, skip_steps: list[Literal['original_object', 'field_images', 
                                                                'angles', 'projection_images', 
                                                                'pre_processed_projection_images',
                                                                'reconstructed_object']]=[],
                    cache: bool=True, reset_pointers: bool=False,
                    skip_angle_resolvers: list[int]=[], skip_pre_processors: list[int]=[],
                    skip_reconstructors: list[int]=[]
                    ) -> None:
        self.log = log
        self.skip_steps = skip_steps
        self.cache = cache #TODO
        self.reset_pointers = reset_pointers
        self.skip_angle_resolvers = skip_angle_resolvers
        self.skip_pre_processors = skip_pre_processors
        self.skip_reconstructors = skip_reconstructors

class SimulatorCtPipeline(AbstractCtPipeline):
    def __init__(self, simulator: SimpleSimulator, motor_controller: SimulatorMotorController, 
                 xray_controller: SimulatorXrayController, xray_setting: SimulatorXraySetting,
                 field_resolver: SimulatorFieldResolver, angle_resolver: list[SimulatorAngleResolver], 
                 image_cache: SimulatorImageCache, image_preprocessor: list[SimulatorImagePreprocessor], 
                 object_reconstructor: list[SimulatorObjectReconstructor], 
                 reconstruction_settings :Optional[SimulatorReconstructionSettings]=None) -> None:
        self.__simulator = simulator
        self.__motor_controller = motor_controller
        self.__xray_controller = xray_controller
        self.__xray_setting = xray_setting
        self.__field_resolver = field_resolver
        self.__angle_resolver = angle_resolver
        self.__image_cache = image_cache
        self.__image_preprocessor = image_preprocessor
        self.__object_reconstructor = object_reconstructor
        if reconstruction_settings is None:
            self.__reconstruction_settings = SimulatorReconstructionSettings(log=True, multiprocessing=True)
        else:
            self.__reconstruction_settings = reconstruction_settings
        self.__reconstruction_settings.set_shape(self.__simulator.object_shape)

    def execute_pipeline(self, pipeline_settings: SimulatorPipelineSettings) -> None:
        # prepare
        [angle_resolver.reset() for angle_resolver in self.__angle_resolver]

        # original object
        if not ('original_object' in pipeline_settings.skip_steps):
            self.__image_cache.cache_data("original_object", self.__simulator.current_object)
        
        # field images
        if not ('field_images' in pipeline_settings.skip_steps):
            flat_field, dark_field = self.__field_resolver.get_field_images()
            self.__image_cache.cache_data("flat_field_images", flat_field)
            self.__image_cache.cache_data("dark_field_images", dark_field)
        
        # angles and projection images
        number_of_angles = self.__angle_resolver[0].number_of_angles
        projection_shape = (number_of_angles, *self.__xray_controller.get_image(self.__xray_setting).shape)
        angle_occupancy_pointer = self.__image_cache.retrieval_pointers['angles'].occupancy_pointer
        if pipeline_settings.log:
            progress_bar_angle_resolver = tqdm(total=len(self.__angle_resolver), desc="Iterating through Angle Resolvers", unit="Angle Resolver", leave=False)
        for i, angle_resolver in enumerate(self.__angle_resolver):
            if i in pipeline_settings.skip_angle_resolvers:
                # move all the pointers
                self.__image_cache.set_pointers_for_ith_angle_resolver(i+1)
                if pipeline_settings.log:
                    progress_bar_angle_resolver.update(1)
                continue
            self.__image_cache.retrieval_pointers['angles'].set_occupancy_pointer(angle_occupancy_pointer)
            if not ('angles' in pipeline_settings.skip_steps):
                angle = angle_resolver.resolve_angle()
            else:
                angles = self.__image_cache.retrieve_data("angles", [len(angle_resolver.angle_history)])
                counter = 0
                angle = angles[counter]
            if pipeline_settings.log:
                progress_bar_angle = tqdm(total=number_of_angles, desc="Scanning Angles", unit="Degrees", leave=False)
            while angle is not None:
                self.__motor_controller.rotate_motor(angle)
                if not ('projection_images' in pipeline_settings.skip_steps):
                    image = self.__xray_controller.get_image(self.__xray_setting)
                    self.__image_cache.cache_data("projection_images", image)
                if not ('angles' in pipeline_settings.skip_steps):
                    angle = angle_resolver.resolve_angle()
                else:
                    if counter < len(angles):
                        counter += 1
                        angle = angles[counter]
                    else:
                        angle = None
                if pipeline_settings.log:
                    progress_bar_angle.update(1)
            if pipeline_settings.log:
                progress_bar_angle.close()
            # pre-processed projection images
            projection_occupancy_pointer = self.__image_cache.retrieval_pointers['projection_images'].occupancy_pointer
            if pipeline_settings.log:
                progress_bar_image_processor = tqdm(total=len(self.__image_preprocessor), desc="Iterating through pre-processors", unit="Image Processor", leave=False)
            for j, image_preprocessor in enumerate(self.__image_preprocessor):
                if j in pipeline_settings.skip_pre_processors:
                    # move all the pointers
                    self.__image_cache.set_pointers_for_ith_pre_processor(j+1)
                    if pipeline_settings.log:
                        progress_bar_image_processor.update(1)
                    continue
                self.__image_cache.retrieval_pointers['projection_images'].set_occupancy_pointer(projection_occupancy_pointer)
                if not ('pre_processed_projection_images' in pipeline_settings.skip_steps):
                    data = self.__image_cache.retrieve_data("projection_images", projection_shape)
                    data = image_preprocessor.preprocess_image(data)
                    self.__image_cache.cache_data("pre_processed_projection_images", data)

                # reconstructed object
                pre_processed_occupancy_pointer = self.__image_cache.retrieval_pointers['pre_processed_projection_images'].occupancy_pointer
                if pipeline_settings.log:
                    progress_bar_object_reconstructor = tqdm(total=len(self.__object_reconstructor), desc="Iterating through reconstructors", unit="Object Reconstructor", leave=False)
                for k, object_reconstructor in enumerate(self.__object_reconstructor):
                    if object_reconstructor in pipeline_settings.skip_reconstructors:
                        # move all the pointers
                        self.__image_cache.set_pointers_for_ith_reconstructor(k)
                        if pipeline_settings.log:
                            progress_bar_object_reconstructor.update(1)
                        continue
                    self.__image_cache.retrieval_pointers['pre_processed_projection_images'].set_occupancy_pointer(pre_processed_occupancy_pointer)
                    if not ('reconstructed_object' in pipeline_settings.skip_steps):
                        data = self.__image_cache.retrieve_data("pre_processed_projection_images", projection_shape)
                        reconstruction = object_reconstructor.reconstruct_object(data, 
                                                                                angle_resolver.angle_history, 
                                                                                self.__reconstruction_settings)
                        self.__image_cache.cache_data("reconstructed_object", reconstruction)
                    if pipeline_settings.log:
                        progress_bar_object_reconstructor.update(1)
                if pipeline_settings.log:
                    progress_bar_object_reconstructor.close()
                    progress_bar_image_processor.update(1)
            if pipeline_settings.log:
                progress_bar_image_processor.close()
                progress_bar_angle_resolver.update(1)
        if pipeline_settings.log:
            progress_bar_angle_resolver.close()
    
    def execute_pipeline_for_whole_dataset(self, pipeline_settings: SimulatorPipelineSettings) -> None:
        if pipeline_settings.log:
            progress_bar = tqdm(total=self.__simulator.number_of_objects, desc="Iterating through Objects", unit="Object", leave=False)
        for batch_i in range(self.__simulator.number_of_objects):
            if batch_i > 0:
                self.__simulator.initialize_new_object()
            self.execute_pipeline(pipeline_settings)
            if pipeline_settings.log:
                progress_bar.update(1)
        if pipeline_settings.log:
            progress_bar.close()
    
    def execute_pipeline_for_batch(self, index: list[int]) -> None:
        for batch_i in index:
            # set the pointers according to the batch_i
            self.__image_cache.set_all_indexes(batch_i)
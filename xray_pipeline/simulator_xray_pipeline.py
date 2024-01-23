import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from xray_pipeline import AbstractXrayPipeline, AbstractPipelineSettings
from simulator import SimpleSimulator
from motor_controller import SimulatorMotorController
from xray_controller import SimulatorXrayController, SimulatorXraySetting
from field_resolver import SimulatorFieldResolver
from angle_resolver import SimulatorAngleResolver
from image_cache import SimulatorImageCache
from image_preprocessor import SimulatorImagePreprocessor
from object_reconstructor import SimulatorObjectReconstructor, SimulatorReconstructionSettings

import numpy as np
from tqdm import tqdm
import cProfile, pstats

class SimulatorPipelineSettings(AbstractPipelineSettings):
    def __init__(self) -> None:
        self.cache = True
        self.log = True

class SimulatorXrayPipeline(AbstractXrayPipeline):
    def __init__(self, simulator: SimpleSimulator, motor_controller: SimulatorMotorController, 
                 xray_controller: SimulatorXrayController, simulator_xray_setting: SimulatorXraySetting,
                 field_resolver: SimulatorFieldResolver, angle_resolver: SimulatorAngleResolver, 
                 image_cache: SimulatorImageCache, image_preprocessor: SimulatorImagePreprocessor, 
                 object_reconstructor: SimulatorObjectReconstructor, 
                 reconstruction_settings:SimulatorReconstructionSettings) -> None:
        self.__simulator = simulator
        self.__motor_controller = motor_controller
        self.__xray_controller = xray_controller
        self.__simulator_xray_setting = simulator_xray_setting
        self.__field_resolver = field_resolver
        self.__angle_resolver = angle_resolver
        self.__image_cache = image_cache
        self.__image_preprocessor = image_preprocessor
        self.__object_reconstructor = object_reconstructor
        self.__reconstruction_settings = reconstruction_settings
    
    def execute_pipeline(self, settings: SimulatorPipelineSettings) -> np.ndarray:
        dark_image, light_image = self.__field_resolver.get_field_images()
        self.__image_cache.cache_field_images(dark_image, light_image)
        is_enough, angle = self.__angle_resolver.resolve_angle()
        if settings.log:
            progress_bar = tqdm(total=180, desc="Getting Xray Images", unit="scanned angle", leave=True)
        while is_enough:
            progress_bar.update(angle - progress_bar.n)
            self.__motor_controller.rotate_motor([angle])
            image = self.__xray_controller.get_image(self.__simulator_xray_setting)
            image = self.__image_preprocessor.preprocess_image(image)
            self.__image_cache.cache_projection_images(image)
            is_enough, angle = self.__angle_resolver.resolve_angle()
        if settings.log:
            progress_bar.update(progress_bar.total - progress_bar.n)
        self.__image_cache.finalize_projection_images()
        reconstruction = self.__object_reconstructor.reconstruct_object(data=self.__image_cache.projection_images[:,0,:,:],
                                                                        angle_history=self.__angle_resolver.angle_history[:-1],
                                                                        reconstruction_settings=self.__reconstruction_settings)
        if settings.cache:
            np.save("pipeline_cache/original_object.npy", self.__simulator.current_objects[0])
            np.save("pipeline_cache/xray_images.npy", self.__image_cache.projection_images)
            np.save("pipeline_cache/angles.npy", np.array(self.__angle_resolver.angle_history[:-1]))
            np.save("pipeline_cache/reconstruction.npy", reconstruction)
        return reconstruction

def main():
    simulator = SimpleSimulator()
    motor_controller = SimulatorMotorController(identification=simulator)
    simulator_xray_setting = SimulatorXraySetting(power='on')
    xray_controller = SimulatorXrayController(identification=simulator)
    field_resolver = SimulatorFieldResolver(xray_controller=xray_controller)
    angle_resolver = SimulatorAngleResolver(delta_theta=1)
    image_cache = SimulatorImageCache()
    image_preprocessor = SimulatorImagePreprocessor()
    object_reconstructor = SimulatorObjectReconstructor()
    reconstruction_settings = SimulatorReconstructionSettings(original_shape=simulator.object_shape)
    pipeline = SimulatorXrayPipeline(simulator=simulator, motor_controller=motor_controller,
                                     xray_controller=xray_controller, simulator_xray_setting=simulator_xray_setting,
                                     field_resolver=field_resolver, angle_resolver=angle_resolver,
                                     image_cache=image_cache, image_preprocessor=image_preprocessor,
                                     object_reconstructor=object_reconstructor,
                                     reconstruction_settings=reconstruction_settings)
    settings = SimulatorPipelineSettings()
    pipeline.execute_pipeline(settings=settings)

if __name__ == "__main__":
    # profile and write to a file
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()
    stats.dump_stats("pipeline_cache/profile.prof")
    
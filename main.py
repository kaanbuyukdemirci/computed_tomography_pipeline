import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from xray_pipeline import SimulatorXrayPipeline, SimulatorPipelineSettings
from simulator import SimpleSimulator
from motor_controller import SimulatorMotorController
from xray_controller import SimulatorXrayController, SimulatorXraySetting
from field_resolver import SimulatorFieldResolver
from angle_resolver import SimulatorAngleResolver
from image_cache import SimulatorImageCache
from image_preprocessor import SimulatorImagePreprocessor
from object_reconstructor import SimulatorObjectReconstructor, SimulatorReconstructionSettings

import cProfile, pstats

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
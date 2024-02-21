import os,sys
import pydicom as dicom
import numpy as np
import cProfile, pstats
from typing import Literal
if True:
    if os.path.abspath(os.getcwd()) not in sys.path:
        sys.path.append(os.getcwd())

from computed_tomography_pipeline import SimpleSimulator, SimulatorMotorController, SimulatorXrayController
from computed_tomography_pipeline import SimulatorXraySetting, SimulatorFieldResolver, SimulatorAngleResolver
from computed_tomography_pipeline import SimulatorImageCache, SimulatorImagePreprocessor, SimulatorObjectReconstructor
from computed_tomography_pipeline import SimpleDataset, SimulatorCtPipeline, SimulatorPipelineSettings
from computed_tomography_pipeline import SimulatorReconstructionSettings

def profiler(function):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = function(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('tottime')
        stats.print_stats()
        stats.dump_stats("profile.prof")
        return result
    return wrapper

def main():
    path = "data/SPHNQA4IQI/"
    cross_section_count = 172
    power = 'on'
    number_of_angles = 32
    big_data_dictionary_path = "big_data_dictionary.hdf5"
    from_saved = False
    skip_steps: list[Literal['original_object', 'field_images', 'angles', 
                             'projection_images', 'pre_processed_projection_images',
                             'reconstructed_object']] = []
    
    simple_dataset = SimpleDataset(path, cross_section_count)
    simple_simulator = SimpleSimulator(iterable_generator_class=simple_dataset)
    number_of_objects = simple_simulator.number_of_objects
    simulator_motor_controller = SimulatorMotorController(identification=simple_simulator)
    simulator_xray_controller = SimulatorXrayController(identification=simple_simulator)
    simulator_xray_setting = SimulatorXraySetting(power=power)
    simulator_field_resolver = SimulatorFieldResolver(xray_controller=simulator_xray_controller)
    simulator_angle_resolver = [SimulatorAngleResolver(number_of_angles=number_of_angles)]
    simulator_image_cache = SimulatorImageCache(big_data_dictionary_path=big_data_dictionary_path,
                                                xray_controller=simulator_xray_controller, 
                                                field_resolver=simulator_field_resolver,
                                                angle_resolver=simulator_angle_resolver, 
                                                number_of_objects=number_of_objects,
                                                from_saved=from_saved)
    simulator_image_preprocessor = [SimulatorImagePreprocessor()]
    simulator_object_reconstructor = [SimulatorObjectReconstructor()]
    simulator_reconstruction_settings = SimulatorReconstructionSettings(log=True, multiprocessing=False)
    simulator_ct_pipeline = SimulatorCtPipeline(simulator=simple_simulator, 
                                                    motor_controller=simulator_motor_controller, 
                                                    xray_controller=simulator_xray_controller, 
                                                    xray_setting=simulator_xray_setting,
                                                    field_resolver=simulator_field_resolver, 
                                                    angle_resolver=simulator_angle_resolver, 
                                                    image_cache=simulator_image_cache, 
                                                    image_preprocessor=simulator_image_preprocessor, 
                                                    object_reconstructor=simulator_object_reconstructor,
                                                    reconstruction_settings=simulator_reconstruction_settings)
    simulator_pipeline_settings = SimulatorPipelineSettings(skip_steps=skip_steps)
    simulator_ct_pipeline.execute_pipeline_for_whole_dataset(pipeline_settings=simulator_pipeline_settings)

if __name__ == "__main__":
    #main()
    profiler(main)()
    pass
   
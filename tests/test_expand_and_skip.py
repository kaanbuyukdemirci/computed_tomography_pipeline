import os,sys
import pydicom as dicom
import numpy as np
import cProfile, pstats, shutil
from typing import Literal
import h5py
if True:
    if os.path.abspath(os.getcwd()) not in sys.path:
        sys.path.append(os.getcwd())

from computed_tomography_pipeline import SimpleSimulator, SimulatorMotorController, SimulatorXrayController
from computed_tomography_pipeline import SimulatorXraySetting, SimulatorFieldResolver, SimulatorAngleResolver
from computed_tomography_pipeline import SimulatorImageCache, SimulatorImagePreprocessor, SimulatorObjectReconstructor
from computed_tomography_pipeline import SimpleDataset, SimulatorCtPipeline, SimulatorPipelineSettings

def main():
    # delete the old one
    if os.path.exists("big_data_dictionary.hdf5"):
        os.remove("big_data_dictionary.hdf5")
    # copy a new one
    shutil.copy("big_data_dictionary_copy.hdf5", "big_data_dictionary.hdf5")

    path = "data/SPHNQA4IQI/"
    cross_section_count = 172
    power = 'on'
    number_of_angles = 32
    big_data_dictionary_path = "big_data_dictionary.hdf5"
    from_saved = True
    number_of_angle_resolvers = 1
    number_of_pre_processors = 1
    number_of_reconstructors = 1
    
    simple_dataset = SimpleDataset(path, cross_section_count)
    simple_simulator = SimpleSimulator(iterable_generator_class=simple_dataset)
    number_of_objects = simple_simulator.number_of_objects
    simulator_motor_controller = SimulatorMotorController(identification=simple_simulator)
    simulator_xray_controller = SimulatorXrayController(identification=simple_simulator)
    simulator_xray_setting = SimulatorXraySetting(power=power)
    simulator_field_resolver = SimulatorFieldResolver(xray_controller=simulator_xray_controller)
    simulator_angle_resolver = [SimulatorAngleResolver(number_of_angles=number_of_angles),
                                SimulatorAngleResolver(number_of_angles=number_of_angles)]
    simulator_image_cache = SimulatorImageCache(None, 
                                                None,
                                                None, 
                                                big_data_dictionary_path=big_data_dictionary_path,
                                                number_of_objects=number_of_objects,
                                                from_saved=from_saved,
                                                number_of_angle_resolvers=number_of_angle_resolvers,
                                                number_of_pre_processors=number_of_pre_processors,
                                                number_of_reconstructors=number_of_reconstructors)

    # check memory
    print("-"*10, "Before:")
    with h5py.File(big_data_dictionary_path, 'r') as f:
        for key in f.keys():
            print(f"{key}: {f[key].shape}")
    
    # expand memory
    number_of_angle_resolvers = 2
    simulator_image_cache.add_angle_resolver(number_of_angle_resolvers)

    # check memory
    print("-"*10, "After:")
    with h5py.File(big_data_dictionary_path, 'r') as f:
        for key in f.keys():
            print(f"{key}: {f[key].shape}")
    print("-"*10)

    # skip first angle resolver
    skip_angle_resolvers = [0]
    skip_pre_processors = []
    skip_reconstructors = []
    skip_steps: list[Literal['original_object', 'field_images', 'angles', 
                             'projection_images', 'pre_processed_projection_images',
                             'reconstructed_object']] = ['original_object', 'field_images']
    simulator_pipeline_settings = SimulatorPipelineSettings(skip_steps=skip_steps, 
                                                            skip_angle_resolvers=skip_angle_resolvers,
                                                            skip_pre_processors=skip_pre_processors,
                                                            skip_reconstructors=skip_reconstructors)

    # complete the pipeline
    simulator_image_preprocessor = [SimulatorImagePreprocessor()]
    simulator_object_reconstructor = [SimulatorObjectReconstructor()]
    simulator_ct_pipeline = SimulatorCtPipeline(simulator=simple_simulator, 
                                                    motor_controller=simulator_motor_controller, 
                                                    xray_controller=simulator_xray_controller, 
                                                    xray_setting=simulator_xray_setting,
                                                    field_resolver=simulator_field_resolver, 
                                                    angle_resolver=simulator_angle_resolver, 
                                                    image_cache=simulator_image_cache, 
                                                    image_preprocessor=simulator_image_preprocessor, 
                                                    object_reconstructor=simulator_object_reconstructor)
    simulator_ct_pipeline.execute_pipeline_for_whole_dataset(pipeline_settings=simulator_pipeline_settings)

if __name__ == "__main__":
    main()
    
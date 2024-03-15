import os, sys
import shutil
from typing import Literal
import h5py

from computed_tomography_pipeline import SimulatorImageCache, SimulatorObjectReconstructor, SimulatorCtPipeline, SimulatorPipelineSettings

def main():
    #path to "big_data_dictionary_256.hdf5"
    big_data_dictionary_path = "D:/dataset/big_data_dictionary_256.hdf5" 
    from_saved = True
    
    simple_simulator = None
    simulator_motor_controller = None
    simulator_xray_controller = None
    simulator_xray_setting = None
    simulator_field_resolver = None
    simulator_angle_resolver = [None]
    simulator_image_cache = SimulatorImageCache(big_data_dictionary_path=big_data_dictionary_path,
                                                xray_controller=simulator_xray_controller,
                                                field_resolver=simulator_field_resolver,
                                                angle_resolver=simulator_angle_resolver,
                                                from_saved=from_saved)
    simulator_image_preprocessor = [None] 
    simulator_object_reconstructor = [SimulatorObjectReconstructor(), 
                                      SimulatorObjectReconstructor(reduce=2),
                                      SimulatorObjectReconstructor(reduce=4),
                                      SimulatorObjectReconstructor(reduce=8)]
    simulator_ct_pipeline = SimulatorCtPipeline(simulator=simple_simulator, 
                                                    motor_controller=simulator_motor_controller, 
                                                    xray_controller=simulator_xray_controller, 
                                                    xray_setting=simulator_xray_setting,
                                                    field_resolver=simulator_field_resolver, 
                                                    angle_resolver=simulator_angle_resolver, 
                                                    image_cache=simulator_image_cache, 
                                                    image_preprocessor=simulator_image_preprocessor, 
                                                    object_reconstructor=simulator_object_reconstructor)
    
    # skip first reconstructor, as it is already done
    skip_angle_resolvers = []
    skip_pre_processors = []
    skip_reconstructors = [0]

    # skip every step other than reconstruction, as they are already done
    skip_steps: list[Literal['original_object', 'field_images', 'angles', 
                             'projection_images', 'pre_processed_projection_images',
                             'reconstructed_object']] = ['original_object', 'field_images', 'angles',
                                                        'projection_images', 'pre_processed_projection_images']
    simulator_pipeline_settings = SimulatorPipelineSettings(skip_steps=skip_steps, 
                                                            skip_angle_resolvers=skip_angle_resolvers,
                                                            skip_pre_processors=skip_pre_processors,
                                                            skip_reconstructors=skip_reconstructors)

    # check memory
    print("-"*10, "Before:")
    with h5py.File(big_data_dictionary_path, 'r') as f:
        for key in f.keys():
            print(f"{key}: {f[key].shape}")
    
    # expand memory
    number_of_reconstructors = len(simulator_object_reconstructor)
    simulator_image_cache.add_reconstructor(number_of_reconstructors)

    # check memory again
    print("-"*10, "After:")
    with h5py.File(big_data_dictionary_path, 'r') as f:
        for key in f.keys():
            print(f"{key}: {f[key].shape}")
    print("-"*10)
    
    # we can now run the pipeline
    simulator_ct_pipeline.execute_pipeline_for_whole_dataset(pipeline_settings=simulator_pipeline_settings)

if __name__ == "__main__":
    main()
    
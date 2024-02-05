import os,sys
import pydicom as dicom
import numpy as np
import cProfile, pstats
if True:
    if os.path.abspath(os.getcwd()) not in sys.path:
        sys.path.append(os.getcwd())

from computed_tomography_pipeline import SimpleSimulator, SimulatorMotorController, SimulatorXrayController
from computed_tomography_pipeline import SimulatorXraySetting, SimulatorFieldResolver, SimulatorAngleResolver
from computed_tomography_pipeline import SimulatorImageCache, SimulatorImagePreprocessor, SimulatorObjectReconstructor
from computed_tomography_pipeline import SimulatorReconstructionSettings, SimulatorCtPipeline, SimulatorPipelineSettings

def data_generator_function_1(path="data/"):
    complete_data_paths = [os.path.join(path, data_path) 
                        for data_path in os.listdir(path) 
                        if data_path.endswith(".dcm")]
    sample = dicom.dcmread(complete_data_paths[0])
    shape = (len(complete_data_paths), sample.Rows, sample.Columns)
    pixel_range = [None, None]
    if sample.PixelRepresentation == 0:
        pixel_range[0] = 0
        pixel_range[1] = 2**sample.BitsStored - 1
    else:
        pixel_range[0] = -2**(sample.BitsStored - 1)
        pixel_range[1] = 2**(sample.BitsStored - 1) - 1
    
    data = np.zeros(shape)
    for i, path in enumerate(complete_data_paths):
        data[i] = (dicom.dcmread(path).pixel_array - pixel_range[0]) / (pixel_range[1] - pixel_range[0])
    
    return data

def data_generator_function_2(path="data/SPHNQA4IQI/"):
    for path_i in os.listdir(path):
        path_i = path + path_i
        for path_j in os.listdir(path_i):
            path_j = path_i + "/" + path_j
            if len(os.listdir(path_j)) == 1:
                break
            else:
                print("sample: ", path_j)
                yield data_generator_function_1(path_j)
    yield None

def profiler(function):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = function(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('tottime')
        stats.print_stats()
        stats.dump_stats("ctp_cache/pipeline/profile.prof")
        return result
    return wrapper

def main():
    data_generator = data_generator_function_2()
    simple_simulator = SimpleSimulator(object_generator_function=data_generator)
    simulator_motor_controller = SimulatorMotorController(identification=simple_simulator)
    simulator_xray_controller = SimulatorXrayController(identification=simple_simulator)
    simulator_xray_setting = SimulatorXraySetting(power='on')
    simulator_field_resolver = SimulatorFieldResolver(xray_controller=simulator_xray_controller)
    simulator_angle_resolver = SimulatorAngleResolver(number_of_angles=3)
    simulator_image_cache = SimulatorImageCache(simulator_xray_controller, 
                                                simulator_field_resolver,
                                                simulator_angle_resolver, 
                                                big_data_dictionary_path="big_data_dictionary.hdf5",
                                                number_of_objects=2,
                                                from_saved=False)
    #print(simulator_image_cache.shapes)
    simulator_image_preprocessor = SimulatorImagePreprocessor()
    simulator_object_reconstructor = SimulatorObjectReconstructor()
    simulator_ct_pipeline = SimulatorCtPipeline(simulator=simple_simulator, 
                                                    motor_controller=simulator_motor_controller, 
                                                    xray_controller=simulator_xray_controller, 
                                                    xray_setting=simulator_xray_setting,
                                                    field_resolver=simulator_field_resolver, 
                                                    angle_resolver=simulator_angle_resolver, 
                                                    image_cache=simulator_image_cache, 
                                                    image_preprocessor=simulator_image_preprocessor, 
                                                    object_reconstructor=simulator_object_reconstructor)
    simulator_pipeline_settings = SimulatorPipelineSettings(start_from='beginning')
    simulator_ct_pipeline.execute_pipeline(pipeline_settings=simulator_pipeline_settings)

if __name__ == "__main__":
    main()
    #profiler(main)()
   
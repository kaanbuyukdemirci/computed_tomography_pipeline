import computed_tomography_pipeline as ctp

# path to SPHNQA4IQI
path = "dataset/manifest-1706015986222/CT-Phantom4Radiomics/SPHNQA4IQI/"
cross_section_count = 343
power = 'on'
number_of_angles = 128
big_data_dictionary_path = "big_data_dictionary_256.hdf5"
from_saved = False
skip_steps = []
simple_dataset = ctp.SimpleDataset(path, cross_section_count)
simple_simulator = ctp.SimpleSimulator(iterable_generator_class=simple_dataset)
number_of_objects = simple_simulator.number_of_objects
simulator_motor_controller = ctp.SimulatorMotorController(identification=simple_simulator)
simulator_xray_controller = ctp.SimulatorXrayController(identification=simple_simulator)
simulator_xray_setting = ctp.SimulatorXraySetting(power=power)
simulator_field_resolver = ctp.SimulatorFieldResolver(xray_controller=simulator_xray_controller)
simulator_angle_resolver = [ctp.SimulatorAngleResolver(number_of_angles=number_of_angles)]
simulator_image_cache = ctp.SimulatorImageCache(simulator_xray_controller, 
                                                simulator_field_resolver,
                                                simulator_angle_resolver, 
                                                big_data_dictionary_path=big_data_dictionary_path,
                                                number_of_objects=number_of_objects,
                                                from_saved=from_saved)
simulator_image_preprocessor = [ctp.SimulatorImagePreprocessor()]
simulator_object_reconstructor = [ctp.SimulatorObjectReconstructor()]
simulator_ct_pipeline = ctp.SimulatorCtPipeline(simulator=simple_simulator, 
                                                motor_controller=simulator_motor_controller, 
                                                xray_controller=simulator_xray_controller, 
                                                xray_setting=simulator_xray_setting,
                                                field_resolver=simulator_field_resolver, 
                                                angle_resolver=simulator_angle_resolver, 
                                                image_cache=simulator_image_cache, 
                                                image_preprocessor=simulator_image_preprocessor, 
                                                object_reconstructor=simulator_object_reconstructor)
simulator_pipeline_settings = ctp.SimulatorPipelineSettings(skip_steps=skip_steps)
simulator_ct_pipeline.execute_pipeline_for_whole_dataset(pipeline_settings=simulator_pipeline_settings)
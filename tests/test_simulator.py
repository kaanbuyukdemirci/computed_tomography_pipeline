from computed_tomography_pipeline import SimpleSimulator
import numpy as np
import cv2

def simple_generator():
    square = np.zeros((256, 256, 256))
    square[100:150, 100:150, 100:150] = 1
    while True:
        yield square

simple_simulator = SimpleSimulator(object_generator_function=simple_generator, log=True)

for angle in range(0, 180, 5):
    simple_simulator.current_angle = angle
    xray_projection = simple_simulator.get_xray_projection()
    xray_projection = xray_projection / xray_projection.max()
    cv2.imshow("xray_projection", xray_projection)
    cv2.waitKey(1)  
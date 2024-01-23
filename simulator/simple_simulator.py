import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from simulator.abstract_simulator import AbstractSimulator

import numpy as np
import pydicom as dicom
from tqdm import tqdm
import torch
from torchvision.transforms.functional import rotate

class SimpleSimulator(AbstractSimulator):
    def __init__(self, treed_data_paths:str='data/', current_object_indexes:list[int]=[0], log:bool=True) -> None:
        """The abstract simulator class. This class is used to simulate the x-ray setting of a 3D object. 
        Assumes that the detector and the cross-section of the object are square.

        Parameters
        ----------
        treed_data_paths : str
            The path to the 3D data.
        current_object_indexes : list[int]
            The indexes of the current objects to use.
        """
        self.log = log
        self.__treed_data_paths = treed_data_paths
        self.current_object_indexes = current_object_indexes
        self.current_angles = [0 for _ in range(len(current_object_indexes))]
        self.__current_objects = None
        self.__object_shape = None
        self.__object_resolution = None
        self.__detector_resolution = None
        self.__xray_projection_shape = None
        self.initialize_dataset()
    
    @property
    def current_objects(self) -> np.ndarray:
        """The current objects to use.

        Returns
        -------
        np.ndarray, shape (n_objects, z, y, x)
            The current objects to use. The shape of the objects is (z, y, x).
        """
        return self.__current_objects #np.load(self.__treed_data_paths + "objects.npy")[self.current_object_indexes]
    
    @property
    def object_shape(self) -> tuple[int, int]:
        return self.__object_shape
    
    @property
    def object_resolution(self) -> int:
        return self.__object_resolution
    
    @property
    def detector_resolution(self) -> int:
        return self.__detector_resolution
    
    @property
    def xray_projection_shape(self) -> tuple[int, int, int]:
        return self.__xray_projection_shape
    
    def initialize_dataset(self):
        if os.path.exists(self.__treed_data_paths + "objects.npy"):
            # set other properties
            objects = np.load(self.__treed_data_paths + "objects.npy")
        else:
            # find out the number of images and read the first image to get the shape and pixel range
            complete_data_paths = [os.path.join(self.__treed_data_paths, data_path) 
                                for data_path in os.listdir(self.__treed_data_paths) 
                                if data_path.endswith(".dcm")]
            object_count = len(complete_data_paths)
            sample_image = dicom.dcmread(complete_data_paths[0])
            pixel_range = [None, None]
            if sample_image.PixelRepresentation == 0:
                pixel_range[0] = 0
                pixel_range[1] = 2**sample_image.BitsStored - 1
            else:
                pixel_range[0] = -2**(sample_image.BitsStored - 1)
                pixel_range[1] = 2**(sample_image.BitsStored - 1) - 1
            image_shape = sample_image.pixel_array.shape
            objects = np.zeros((1, object_count, *image_shape))
            
            # read all images
            if self.log:
                bar = tqdm(total=object_count, desc="Reading images", unit="image", leave=False)
            for i, path in enumerate(complete_data_paths):
                objects[0,i] = (dicom.dcmread(path).pixel_array - pixel_range[0]) / (pixel_range[1] - pixel_range[0])
                if self.log:
                    bar.update()
            
            # save all the images as a numpy array
            np.save(self.__treed_data_paths + "objects.npy", objects)
        
        # set other properties
        self.__current_objects = objects[self.current_object_indexes]
        self.__object_shape = objects.shape[1:]
        self.__object_resolution = max(objects.shape[-2:]) # assumes square images
        res = np.ceil(self.object_resolution * np.sqrt(2)).astype(int)
        if (res % 2 == 1) and (self.object_resolution % 2 == 0): res += 1
        elif (res % 2 == 0) and (self.object_resolution % 2 == 1): res += 1
        self.__detector_resolution = res
        self.__xray_projection_shape = (len(self.current_angles), self.detector_resolution, self.detector_resolution)
        
    def get_xray_projection(self) -> np.ndarray:
        """Get the x-ray projection of the current image at the given angle (self.current_angles).
        The motor rotates around the z-axis, and the x-ray is projected through the y-axis.
        
        Returns
        -------
        np.ndarray
            The x-ray projection of the current objects at given angles.
        """
        # prepare
        objects = torch.from_numpy(self.current_objects)
        angles = self.current_angles
        detector_resolution = self.detector_resolution
        resolution_difference = detector_resolution - self.object_resolution
        pad_size = int(resolution_difference / 2)
        objects_shape = objects.shape[:2]
        rotated_objects = torch.zeros(*objects_shape, detector_resolution, detector_resolution)
        rotated_objects[:, :, pad_size:-pad_size, pad_size:-pad_size] = objects
        del objects
        
        # rotate (- for clockwise, + for counter-clockwise)
        if all([angle == angles[0] for angle in angles]):
            rotated_objects = rotate(rotated_objects, -angles[0])
        else:
            for i, object in enumerate(rotated_objects):
                rotated_objects[i] = rotate(object, -angles[i])
        
        # project
        rotated_objects = rotated_objects.numpy()
        if False:
            print()
            image = rotated_objects[0,100]
            plot = rotated_objects.sum(axis=2)[0,100]
            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap="gray")
            plt.subplot(1, 2, 2)
            plt.plot(plot, np.arange(plot.shape[0]))
            plt.show()
            quit()
        rotated_objects = rotated_objects.sum(axis=2)
        return rotated_objects

def main():
    import cv2
    import matplotlib.pyplot as plt
    render = False
    #simulator = SimpleSimulator('pipeline_cache/')
    simulator = SimpleSimulator()
    if render:
        max_val = 0
        for theta in range(0, 91, 5):
            simulator.current_angles = [theta]
            image = simulator.get_xray_projection()[0].transpose(1, 0)
            max_val = max(image.max(), max_val)
            cv2.putText(image, f"theta: {theta}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("image", image/max_val)
            cv2.waitKey(100)
    else:
        simulator.current_angles = [45]
        image = simulator.get_xray_projection()[0].transpose(1, 0)
        plt.imshow(image, cmap="gray")
        plt.title(image.shape)
        plt.show()

if __name__ == "__main__":
    main()
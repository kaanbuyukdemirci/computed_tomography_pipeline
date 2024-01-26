from abc import ABC, abstractmethod
import numpy as np

class AbstractSimulator(ABC):
    def __init__(self, treed_data_paths:str, current_object_indexes:list[int]) -> None:
        """The abstract simulator class. This class is used to simulate the x-ray setting of a 3D object. 

        Parameters
        ----------
        treed_object_path : str
            The path to the 3D data.
        current_object_indexes : list[int]
            The indexes of the current objects to use.
        """
        super().__init__()
        self.__treed_data_paths = treed_data_paths
        self.current_object_indexes = current_object_indexes
        self.angles = [0 for _ in range(len(current_object_indexes))]
    
    @property
    @abstractmethod
    def current_objects(self) -> np.ndarray:
        """The current objects to use.

        Returns
        -------
        np.ndarray, shape (n_objects, z, y, x)
            The current objects to use. The shape of the objects is (z, y, x).
        """
        pass
    
    @property
    @abstractmethod
    def xray_projection_shape(self) -> tuple[int, int, int]:
        """The shape of the x-ray projection.

        Returns
        -------
        tuple[int, int, int]
            The shape of the x-ray projection.
        """
        pass
    
    @abstractmethod
    def get_xray_projection(self) -> np.ndarray:
        """Get the x-ray projection of the current image at the given angle (self.current_angles).
        The motor rotates around the z-axis, and the x-ray is projected through the y-axis.
        
        Returns
        -------
        np.ndarray
            The x-ray projection of the current objects at given angles.
        """
        pass
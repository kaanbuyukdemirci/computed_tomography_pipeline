import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import tqdm
import time
import torch
import cv2

class CTReconstructionDataset(Dataset):
    def __init__(self, data_path, neighbor_count=3):
        self.data_path = data_path
        self.primary_key = "reconstructed_object"
        self.secondary_keys = ["angles", "original_object"]
        self.neighbor_count = neighbor_count
        self.valid_cross_section_interval = [22, 220]
        with h5py.File(self.data_path, 'r') as h5f:
            shape = h5f[self.primary_key].shape
            self.__object_count = shape[0]
            self.__cross_section_count = self.valid_cross_section_interval[1] - self.valid_cross_section_interval[0]
            self.__length = self.__object_count * self.__cross_section_count
            
        self.keys = [self.primary_key] + self.secondary_keys

    def __len__(self):
        return self.__length

    def __getitem__(self, idx):
        object_index = idx // self.__cross_section_count
        cross_section_index = idx % self.__cross_section_count
        sample = {}
        with h5py.File(self.data_path, 'r') as h5f:
            sample["reconstructed_cross_section"] = h5f[self.primary_key][object_index, :, :, :, cross_section_index]
            counter = 0
            unique_neighbor_indexes = []
            neighbor_indexes = []
            for i in range(-self.neighbor_count, self.neighbor_count+1):
                index = cross_section_index + i
                if index < 0:
                    neighbor_indexes.append(counter)
                elif index >= self.__cross_section_count:
                    neighbor_indexes.append(counter-1)
                else:
                    neighbor_indexes.append(counter)
                    unique_neighbor_indexes.append(index + self.valid_cross_section_interval[0])
                    counter += 1
            #print(neighbor_indexes, unique_neighbor_indexes)
            # angles
            sample["angles"] = h5f["angles"][object_index]

            # original data
            unique_data = h5f["original_object"][object_index, unique_neighbor_indexes]
            #sample["original_cross_section_neighborhood"] = unique_data[neighbor_indexes]
            sample["original_cross_section"] = unique_data[neighbor_indexes[self.neighbor_count]]
            
            # reconstructed data
            unique_data = h5f[self.primary_key][object_index, :, :, :, unique_neighbor_indexes]
            sample["reconstructed_cross_section_neighborhood"] = unique_data[:, :, :, neighbor_indexes]
            sample["reconstructed_cross_section"] = unique_data[:, :, :, neighbor_indexes[self.neighbor_count]]

        return sample
        
def test_shapes():
    data_path = "D:/dataset/big_data_dictionary_256_128_64_32.hdf5"
    dataset = CTReconstructionDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    print(len(dataloader))
    for batch in dataloader:
        for key in batch:
            print(key, batch[key].shape, batch[key].dtype)
        break

def test_random_split():
    data_path = "D:/dataset/big_data_dictionary_256_128_64_32.hdf5"
    dataset = CTReconstructionDataset(data_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    print(len(train_dataset), len(test_dataset), type(train_dataset), type(test_dataset))

def test_neighborhood():
    data_path = "D:/dataset/big_data_dictionary_256_128_64_32.hdf5"
    dataset = CTReconstructionDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for batch in dataloader:
        neighborhood = batch["reconstructed_cross_section_neighborhood"][0, 0, 0, 0]
        for i, image in enumerate(neighborhood):
            text = f"image {i}"
            image = image.numpy()
            image = (image * 255).astype(np.uint8)
            image = cv2.putText(image, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) 
            cv2.imshow("image", image)
            cv2.waitKey(100)

def test_read_all(num_workers=0):
    data_path = "D:/dataset/big_data_dictionary_256_128_64_32.hdf5"
    dataset = CTReconstructionDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=num_workers)
    length = len(dataloader)
    bar = tqdm.tqdm(total=length)
    for batch in dataloader:
        bar.update(len(batch["reconstructed_cross_section_neighborhood"]))

def test_bias():
    valid_cross_section_interval = [22, 220]
    path = "D:/dataset/big_data_dictionary_256_128_64_32.hdf5"
    with h5py.File(path, 'r') as h5f:
        reconstructed_object = h5f["original_object"]
        for i in range(-3, 4):
            index = np.random.randint(0, reconstructed_object.shape[0])
            image = reconstructed_object[index]#, 0, 0, 0]
            image1 = image[valid_cross_section_interval[0]+i]
            image2 = image[valid_cross_section_interval[1]+i]
            text = f"image {i}"
            #image = np.concatenate([image1, image2], axis=1)
            image1 = (image1 * 255).astype(np.uint8)
            image1 = cv2.putText(image1, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            image2 = (image2 * 255).astype(np.uint8)
            image2 = cv2.putText(image2, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("image1", image1)
            cv2.imshow("image2", image2)
            cv2.waitKey()

def test_images():
    data_path = "D:/dataset/big_data_dictionary_256_128_64_32.hdf5"
    dataset = CTReconstructionDataset(data_path, neighbor_count=1)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for batch in dataloader:
        neighborhood = batch["reconstructed_cross_section_neighborhood"][0, 0, 0, 2].numpy()
        neighborhood = np.concatenate(neighborhood, axis=1)
        reconstructed = batch["reconstructed_cross_section"][0, 0, 0, 2].numpy()
        original = batch["original_cross_section"][0].numpy()
        image = np.concatenate([reconstructed, original], axis=1)
        value_range = np.max(image), np.min(image)
        image = (image * 255).astype(np.uint8)
        text = f"value range: {value_range}"
        image = cv2.putText(image, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("image", image)
        cv2.imshow("neighborhood", neighborhood)
        key = cv2.waitKey()
        if key == 'q':
            break
        

if __name__ == "__main__":
    test_images()
    #test_read_all(num_workers=6)

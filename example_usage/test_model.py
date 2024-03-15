import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

from custom_datasets import CTReconstructionDataset
from example_usage.train_model import YourLightningModule
from example_usage.cnn_model import Model as DeepConvNet

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# load the model from a checkpoint and hparams.yaml file
path = "lightning/mymodel/version_15/checkpoints/epoch=9-step=26289.ckpt"
#hyper_parameter_path = "lightning/mymodel/version_0/hparams.yaml"
criteria = torch.nn.MSELoss()
model = DeepConvNet(neighbor_count=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#model = YourLightningModule.load_from_checkpoint(path, model=model, criterion=criteria, optimizer=optimizer, train_index=0, test_index=1)
model = YourLightningModule.load_from_checkpoint(path, model=model, criterion=criteria)

# load the test dataset
data_path = "D:/dataset/big_data_dictionary_256_128_64_32.hdf5"
neighbor_count = model.model.neighbor_count
test_index = model.test_index
test_dataset = CTReconstructionDataset(data_path, neighbor_count=neighbor_count)
test_dataset = torch.utils.data.Subset(test_dataset, test_index)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# test the model
#trainer = pl.Trainer()
#trainer.test(model, test_dataloaders=test_dataloader)

# test the model using visualization
for batch in test_dataloader:
    # reconstruction_64, reconstruction_256, enhanced_64, original_cross_section
    reconstruction_64 = batch['reconstructed_cross_section_neighborhood'][:, 0, 0, 2].cuda()
    reconstruction_256 = batch['reconstructed_cross_section_neighborhood'][:, 0, 0, 0].cuda()
    original_cross_section = batch['original_cross_section'].cuda()
    enhanced_64 = model(reconstruction_64)

    reconstruction_64 = reconstruction_64.squeeze().detach().cpu().numpy()[neighbor_count]
    reconstruction_256 = reconstruction_256.squeeze().detach().cpu().numpy()[neighbor_count]
    original_cross_section = original_cross_section.squeeze().detach().cpu().numpy()
    enhanced_64 = enhanced_64.squeeze().detach().cpu().numpy()

    print(reconstruction_64.shape, reconstruction_256.shape, enhanced_64.shape, original_cross_section.shape)

    # calculate psnr and ssim
    reconstruction_64_ssim = psnr(reconstruction_64, original_cross_section), ssim(reconstruction_64, original_cross_section, data_range=1)
    reconstruction_256_ssim = psnr(reconstruction_256, original_cross_section), ssim(reconstruction_256, original_cross_section, data_range=1)
    enhanced_64_ssim = psnr(enhanced_64, original_cross_section), ssim(enhanced_64, original_cross_section, data_range=1)
    
    # image
    image_row_1 = np.concatenate([reconstruction_64, reconstruction_256], axis=1)
    image_row_2 = np.concatenate([enhanced_64, original_cross_section], axis=1)

    #image = outputs
    reconstruction_64_ssim_text = f"reconstruction_64: psnr: {round(reconstruction_64_ssim[0], 3)}, ssim: {round(reconstruction_64_ssim[1], 3)}"
    reconstruction_256_ssim_text = f"reconstruction_256: psnr: {round(reconstruction_256_ssim[0], 3)}, ssim: {round(reconstruction_256_ssim[1], 3)}"
    enhanced_64_ssim_text = f"enhanced_64: psnr: {round(enhanced_64_ssim[0], 3)}, ssim: {round(enhanced_64_ssim[1], 3)}"
    text = reconstruction_64_ssim_text + reconstruction_256_ssim_text + enhanced_64_ssim_text
    text_1 = reconstruction_64_ssim_text + "-----" + reconstruction_256_ssim_text
    text_2 = enhanced_64_ssim_text

    # normalize
    image_row_1 = (image_row_1 - np.min(image_row_1)) / (np.max(image_row_1) - np.min(image_row_1))
    image_row_2 = (image_row_2 - np.min(image_row_2)) / (np.max(image_row_2) - np.min(image_row_2))

    # convert to uint8
    image_row_1 = (image_row_1 * 255).astype(np.uint8)
    image_row_2 = (image_row_2 * 255).astype(np.uint8)

    # put text
    image_row_1 = cv2.putText(image_row_1, reconstruction_64_ssim_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    image_row_1 = cv2.putText(image_row_1, reconstruction_256_ssim_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    image_row_2 = cv2.putText(image_row_2, text_2, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # save and show
    #cv2.imwrite('64_vs_256.png', image_row_1)
    #cv2.imwrite('ai_vs_original.png', image_row_2)
    cv2.imshow('64 vs 256', image_row_1)
    cv2.imshow('ai vs original', image_row_2)

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    #break
    if key == 'q':
        break
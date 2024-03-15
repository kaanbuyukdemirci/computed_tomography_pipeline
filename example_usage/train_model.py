import pytorch_lightning as pl
#from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler, SimpleProfiler, PassThroughProfiler, Profiler
import wandb
import torch
from torch.utils.data import DataLoader
import numpy as np
import os

from example_usage.cnn_model import Model as DeepConvNet
from custom_datasets import CTReconstructionDataset

class YourLightningModule(pl.LightningModule):
    def __init__(self, model, criterion, optimizer, train_index, validation_index, test_index):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_index = train_index
        self.validation_index = validation_index
        self.test_index = test_index
        self.angle_resolver_index = 0
        self.preprocessor_index = 0
        self.reconstructor_index = 2
        self.save_hyperparameters(ignore=['model', 'criterion'])
        self.example_input_array = torch.randn(2, self.model.neighborhood_size, 512, 512)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model.train()
        inputs = batch['reconstructed_cross_section_neighborhood'][:, self.angle_resolver_index, self.preprocessor_index, self.reconstructor_index]
        labels = batch['original_cross_section']
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log('input_mean', inputs.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log('output_mean', outputs.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        inputs = batch['reconstructed_cross_section_neighborhood'][:, self.angle_resolver_index, self.preprocessor_index, self.reconstructor_index]
        labels = batch['original_cross_section']
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        self.model.eval()
        inputs = batch['reconstructed_cross_section_neighborhood'][:, self.angle_resolver_index, self.preprocessor_index, self.reconstructor_index]
        labels = batch['original_cross_section']
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss

    def inference_step(self, batch, batch_idx):
        inputs = batch['reconstructed_cross_section_neighborhood'][:, self.angle_resolver_index, self.preprocessor_index, self.reconstructor_index]
        outputs = self(inputs)
        outputs = torch.clamp(outputs, 0, 1)
        outputs = outputs.squeeze().detach().cpu().numpy()
        return outputs

    def configure_optimizers(self):
        return self.optimizer

def main():
    # parameters
    neighbor_count = 3

    # model
    model = DeepConvNet(neighbor_count=neighbor_count)

    # criterion
    criterion = torch.nn.MSELoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Bring in your data
    data_path = "D:/dataset/big_data_dictionary_256_128_64_32.hdf5"
    dataset = CTReconstructionDataset(data_path, neighbor_count=neighbor_count)
    train_size = int(0.90 * len(dataset))
    validation_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - validation_size
    train_dataset_index  = np.random.choice(len(dataset), train_size, replace=False)
    validation_dataset_index  = np.random.choice(np.setdiff1d(np.arange(len(dataset)), train_dataset_index), validation_size, replace=False)
    test_dataset_index  = np.setdiff1d(np.arange(len(dataset)), np.concatenate([train_dataset_index, validation_dataset_index]))

    # Initialize the LightningModule and LightningDataModule
    model = YourLightningModule(model, criterion, optimizer, train_dataset_index, validation_dataset_index, test_dataset_index)

    # Initialize the DataLoader
    train_index = model.train_index
    validation_index = model.test_index
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    validation_dataset = torch.utils.data.Subset(dataset, validation_index)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=torch.get_num_threads(),  persistent_workers=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=torch.get_num_threads(), persistent_workers=True)

    # Train the model using a Trainer
    # also log to tensorboard and save all the arguments
    path = 'lightning'
    name = 'mymodel'
    # version = last version + 1
    versions = [int(x.split('_')[-1]) for x in os.listdir(path + '/' + name)]
    version = max(versions) + 1 if len(versions) > 0 else 0
    logger_1 = TensorBoardLogger(path, name=name, version=version, default_hp_metric=False, log_graph=True)
    #logger_2 = WandbLogger(name=name, project='test')
    #profiler = PyTorchProfiler()
    #trainer = pl.Trainer(logger=[logger_1, logger_2], max_epochs=1, log_every_n_steps=40, profiler=profiler)
    trainer = pl.Trainer(logger=logger_1, max_epochs=10, log_every_n_steps=40, val_check_interval=0.25)
    trainer.fit(model, train_dataloader, validation_dataloader)

if __name__ == "__main__":
    main()
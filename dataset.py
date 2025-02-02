import torch
import h5py
import numpy as np
from one_hot_encoder_decoder import one_hot_encode_land_use

class BigEarthNetDataSet(torch.utils.data.Dataset):
    def __init__(self, split):
        cache_file = f"cache/{split}.h5"
        # Open the HDF5 file in read mode
        self.h5_file = h5py.File(cache_file, 'r')
        
        # Get the datasets
        self.images = self.h5_file['images']
        self.labels = self.h5_file['labels']
        
        # Store metadata for reference if needed
        self.metadata_group = self.h5_file['metadata']

    def __getitem__(self, idx):
        # Load image data and convert from uint8 to normalized float32
        image_data = self.images[idx]
        image_data = torch.from_numpy(image_data).float() / 255.0
        
        # Load pre-computed labels
        label = torch.from_numpy(self.labels[idx]).float()
        
        return image_data, label

    def __len__(self):
        return len(self.images)
    
    def __del__(self):
        # Ensure the HDF5 file is properly closed
        try:
            self.h5_file.close()
        except:
            pass
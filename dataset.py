import torch
import os
import pandas as pd
import glob
import rasterio
import numpy as np
from one_hot_encoder_decoder import one_hot_encode_land_use

class BigEarthNetDataSet(torch.utils.data.Dataset):
    def __init__(self, split, image_directory, metadata_file):
        self.image_directory = image_directory
        df = pd.read_parquet(metadata_file)
        self.metadata_df = df[df['split'] == split].reset_index(drop=True)
        
        # Pre-compute file paths
        self.image_paths = {}
        for idx, row in self.metadata_df.iterrows():
            patch_id = row.patch_id
            dir_path = self.__get_image_directory_from_patch_id(image_directory, patch_id)
            self.image_paths[idx] = {
                'B02': glob.glob(os.path.join(dir_path, '*B02.tif'))[0],
                'B03': glob.glob(os.path.join(dir_path, '*B03.tif'))[0],
                'B04': glob.glob(os.path.join(dir_path, '*B04.tif'))[0]
            }

    def __getitem__(self, idx):
        metadata = self.metadata_df.iloc[idx]  
        labels = metadata.labels
        encoded_labels = one_hot_encode_land_use(labels)
        
        paths = self.image_paths[idx]
        image_data = self.__get_rgb_image_data(paths)
        
        return image_data, encoded_labels

    def __len__(self):
        return len(self.metadata_df)

    def __get_rgb_image_data(self, paths):
        # Read all bands at once using memory mapping
        bands = []
        for band_path in [paths['B04'], paths['B03'], paths['B02']]:
            with rasterio.open(band_path) as src:
                band = src.read(1, out_dtype=np.float32) 
                bands.append(torch.from_numpy(band))
        
        rgb = torch.stack(bands) 
        
        # Standardize normalization
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        return rgb

    def __get_image_directory_from_patch_id(self, base_path, string_id):
        dir_name = '_'.join(string_id.split('_')[:-2])
        return os.path.join(base_path, dir_name, string_id)
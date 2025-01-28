#!/usr/bin/env python3

import torch
import os
from pathlib import Path
import pandas as pd
import glob
import rasterio
import numpy as np
from one_hot_encoder_decoder import one_hot_encode_land_use
import argparse
from tqdm import tqdm
import logging
from pathlib import Path
import h5py


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_dataset(image_directory, metadata_file, cache_file, split):
    """
    Preprocess the dataset and save to a single numpy file.
    
    Args:
        image_directory (str): Path to the directory containing image folders
        metadata_file (str): Path to the metadata parquet file
        cache_file (str): Path where the cached numpy file will be saved
        split (str): Dataset split to process ('train', 'validation', or 'test')
    """
    logger.info(f"Loading metadata from {metadata_file}")
    df = pd.read_parquet(metadata_file)
    metadata_df = df[df['split'] == split].reset_index(drop=True)
    
    num_samples = len(metadata_df)
    logger.info(f"Found {num_samples} samples for split '{split}'")
    
    # Get the number of classes from the first sample's labels
    first_sample_labels = metadata_df.iloc[0]['labels']
    num_classes = len(one_hot_encode_land_use(first_sample_labels))
    
    # Pre-allocate arrays with reduced memory footprint
    image_data = np.zeros((num_samples, 3, 120, 120), dtype=np.uint8)
    labels = np.zeros((num_samples, num_classes), dtype=np.float32)  # keep labels as float32
    
    # Create progress bar
    pbar = tqdm(total=num_samples, desc="Processing images")
    
    for idx, row in metadata_df.iterrows():
        patch_id = row['patch_id']
        dir_path = os.path.join(image_directory, '_'.join(patch_id.split('_')[:-2]), patch_id)
        
        try:
            # Get file paths
            paths = {
                'B02': glob.glob(os.path.join(dir_path, '*B02.tif'))[0],
                'B03': glob.glob(os.path.join(dir_path, '*B03.tif'))[0],
                'B04': glob.glob(os.path.join(dir_path, '*B04.tif'))[0]
            }
            
            # Read and process bands
            bands = []
            for band_path in [paths['B04'], paths['B03'], paths['B02']]:
                with rasterio.open(band_path) as src:
                    band = src.read(1).astype(np.float32)
                    bands.append(band)
            
            rgb = np.stack(bands)
            # Normalize to 0-1 and then scale to 0-255 for uint8
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            rgb = (rgb * 255).astype(np.uint8)
            
            # Store processed data
            image_data[idx] = rgb
            labels[idx] = one_hot_encode_land_use(row['labels'])
            
        except Exception as e:
            logger.error(f"Error processing patch {patch_id}: {str(e)}")
            continue
        
        pbar.update(1)
    
    pbar.close()
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    logger.info(f"Saving cached data to {cache_file}")
    with h5py.File(cache_file, 'w') as f:
        f.create_dataset('images', data=image_data)
        f.create_dataset('labels', data=labels)
        
        # For the metadata dictionary, we need to handle it differently
        # since HDF5 doesn't directly store Python dictionaries
        metadata_group = f.create_group('metadata')
        for i, record in enumerate(metadata_df.to_dict('records')):
            record_group = metadata_group.create_group(str(i))
            for key, value in record.items():
                record_group.attrs[key] = value
    logger.info("Preprocessing complete!")

class CachedBigEarthNetDataSet(torch.utils.data.Dataset):
    def __init__(self, cache_file):
        self.data = np.load(cache_file, mmap_mode='r')
        self.images = self.data['images']
        self.labels = self.data['labels']
        
    def __getitem__(self, idx):
        # Convert uint8 back to float32 normalized 0-1
        image = torch.from_numpy(self.images[idx].copy()).float() / 255.0
        label = torch.from_numpy(self.labels[idx].copy())
        return image, label
    
    def __len__(self):
        return len(self.images)

def main():
    parser = argparse.ArgumentParser(description='Preprocess BigEarthNet dataset and create cached files')
    parser.add_argument('--image-dir', required=True, help='Directory containing BigEarthNet images')
    parser.add_argument('--metadata', required=True, help='Path to metadata parquet file')
    parser.add_argument('--output-dir', required=True, help='Directory to save cached files')
    parser.add_argument('--splits', nargs='+', default=['test', 'validation', 'train'],
                      help='Dataset splits to process (default: train validation test)')
    parser.add_argument('--log-file', help='Path to log file (optional)')
    
    args = parser.parse_args()
    
    # Add file handler if log file is specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    # Process each split
    for split in args.splits:
        cache_file = os.path.join(args.output_dir, f'{split}.h5')
        if(Path(cache_file).exists()):
            logger.info(f"Skipping {split} split, file found")
            continue

        logger.info(f"Processing {split} split...")
        
        preprocess_dataset(
            image_directory=args.image_dir,
            metadata_file=args.metadata,
            cache_file=cache_file,
            split=split
        )

if __name__ == "__main__":
    main()
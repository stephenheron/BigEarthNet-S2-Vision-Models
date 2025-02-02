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
    image_data = np.zeros((num_samples, 5, 120, 120), dtype=np.uint8)
    labels = np.zeros((num_samples, num_classes), dtype=np.uint8) 
    
    # Create progress bar
    pbar = tqdm(total=num_samples, desc="Processing images")
    
    for idx, row in metadata_df.iterrows():
        patch_id = row['patch_id']
        dir_path = os.path.join(image_directory, '_'.join(patch_id.split('_')[:-2]), patch_id)
        
        try:
            # Get file paths
            paths = {
                'B02': glob.glob(os.path.join(dir_path, '*B02.tif'))[0],  # Blue
                'B03': glob.glob(os.path.join(dir_path, '*B03.tif'))[0],  # Green
                'B04': glob.glob(os.path.join(dir_path, '*B04.tif'))[0],  # Red
                'B08': glob.glob(os.path.join(dir_path, '*B08.tif'))[0],  # NIR
                'B11': glob.glob(os.path.join(dir_path, '*B11.tif'))[0]   # SWIR 1
            }

            # Read and process bands
            bands = []
            for band_path in [paths['B04'], paths['B03'], paths['B02'], paths['B08']]:
                with rasterio.open(band_path) as src:
                    band = src.read(1).astype(np.float32)
                    bands.append(band)

            # Handle SWIR band separately with upsampling
            with rasterio.open(paths['B11']) as src:
                # Get the target shape from one of the 10m bands
                with rasterio.open(paths['B04']) as ref:
                    target_shape = ref.read(1).shape
                
                # Resample SWIR to 10m
                swir = src.read(
                    1,
                    out_shape=target_shape,
                    resampling=rasterio.enums.Resampling.bilinear
                ).astype(np.float32)
                bands.append(swir)

            # Stack all bands
            all_bands = np.stack(bands)

            normalized_bands = np.zeros_like(all_bands)
            eps = 1e-8  # Small number to prevent division by zero
            for i in range(all_bands.shape[0]):
                band = all_bands[i]
                band_min = band.min()  # Calculate min for this band
                band_max = band.max()  # Calculate max for this band
                normalized_bands[i] = (band - band_min) / (band_max - band_min + eps)

            all_bands = (normalized_bands * 255).astype(np.uint8) 
            
            # Store processed data
            image_data[idx] = all_bands
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
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
import h5py

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def rotate_bands(bands, angle):
    """Rotate the stacked bands by the specified angle"""
    if angle not in [90, 180, 270]:
        raise ValueError("Angle must be 90, 180, or 270 degrees")
    k = angle // 90
    return np.rot90(bands, k=k, axes=(1, 2))

def flip_bands(bands, direction='horizontal'):
    """Flip the stacked bands horizontally or vertically"""
    if direction not in ['horizontal', 'vertical']:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")
    if direction == 'horizontal':
        return np.flip(bands, axis=2)
    else:
        return np.flip(bands, axis=1)

def transform_bands(bands, rotation=None, flip_direction=None):
    """Apply rotation and/or flipping to the stacked bands"""
    transformed = bands.copy()
    if rotation is not None:
        transformed = rotate_bands(transformed, rotation)
    if flip_direction is not None:
        transformed = flip_bands(transformed, flip_direction)
    return transformed

def preprocess_dataset(image_directory, metadata_file, cache_file, split, batch_size=3000):
    """Preprocess the dataset in batches and save to HDF5 file"""
    logger.info(f"Loading metadata from {metadata_file}")
    df = pd.read_parquet(metadata_file)
    metadata_df = df[df['split'] == split].reset_index(drop=True)
    
    is_training = split == 'train'
    augmentation_factor = 6 if is_training else 1
    num_samples = len(metadata_df) * augmentation_factor
    
    logger.info(f"Found {len(metadata_df)} original samples for split '{split}'")
    if is_training:
        logger.info(f"Will create {num_samples} samples after augmentation")
    
    # Get the number of classes from the first sample's labels
    first_sample_labels = metadata_df.iloc[0]['labels']
    num_classes = len(one_hot_encode_land_use(first_sample_labels))
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    # Create the HDF5 file and initialize datasets
    with h5py.File(cache_file, 'w') as f:
        # Create datasets
        images_dataset = f.create_dataset('images', 
                                        shape=(num_samples, 5, 120, 120),
                                        dtype=np.uint8)
        
        labels_dataset = f.create_dataset('labels',
                                        shape=(num_samples, num_classes),
                                        dtype=np.uint8)
        
        metadata_group = f.create_group('metadata')
        current_idx = 0
        
        # Process data in batches
        batch_images = []
        batch_labels = []
        batch_metadata = []
        
        pbar = tqdm(total=num_samples, desc=f"Processing {split} split")
        
        for idx, row in metadata_df.iterrows():
            try:
                patch_id = row['patch_id']
                dir_path = os.path.join(image_directory, '_'.join(patch_id.split('_')[:-2]), patch_id)
                
                # Get file paths
                paths = {
                    'B02': glob.glob(os.path.join(dir_path, '*B02.tif'))[0],
                    'B03': glob.glob(os.path.join(dir_path, '*B03.tif'))[0],
                    'B04': glob.glob(os.path.join(dir_path, '*B04.tif'))[0],
                    'B08': glob.glob(os.path.join(dir_path, '*B08.tif'))[0],
                    'B11': glob.glob(os.path.join(dir_path, '*B11.tif'))[0]
                }

                # Process bands
                bands = []
                for band_path in [paths['B04'], paths['B03'], paths['B02'], paths['B08']]:
                    with rasterio.open(band_path) as src:
                        band = src.read(1).astype(np.float32)
                        bands.append(band)

                # Handle SWIR band
                with rasterio.open(paths['B11']) as src:
                    with rasterio.open(paths['B04']) as ref:
                        target_shape = ref.read(1).shape
                    swir = src.read(1, out_shape=target_shape,
                                  resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
                    bands.append(swir)

                # Stack and normalize
                all_bands = np.stack(bands)
                normalized_bands = np.zeros_like(all_bands)
                eps = 1e-8
                
                for i in range(all_bands.shape[0]):
                    band = all_bands[i]
                    band_min = band.min()
                    band_max = band.max()
                    normalized_bands[i] = (band - band_min) / (band_max - band_min + eps)

                all_bands = (normalized_bands * 255).astype(np.uint8)
                
                augmentations = [(None, None)] if not is_training else [
                    (None, None),           # Original
                    (90, None),             # 90-degree rotation
                    (180, None),            # 180-degree rotation
                    (270, None),            # 270-degree rotation
                    (None, 'horizontal'),   # Horizontal flip
                    (None, 'vertical')      # Vertical flip
                ]
                
                for rotation, flip_direction in augmentations:
                    transformed = transform_bands(all_bands, rotation, flip_direction)
                    batch_images.append(transformed)
                    batch_labels.append(one_hot_encode_land_use(row['labels']))
                    
                    metadata = row.to_dict()
                    metadata['augmentation'] = 'original'
                    if rotation is not None:
                        metadata['augmentation'] = f'rotate_{rotation}'
                    elif flip_direction is not None:
                        metadata['augmentation'] = f'flip_{flip_direction}'
                    batch_metadata.append(metadata)
                
                pbar.update(len(augmentations))
                
                # Write batch to disk when it reaches batch_size
                if len(batch_images) >= batch_size or idx == len(metadata_df) - 1:
                    batch_size = len(batch_images)
                    
                    # Write images and labels
                    images_dataset[current_idx:current_idx + batch_size] = np.stack(batch_images)
                    labels_dataset[current_idx:current_idx + batch_size] = np.stack(batch_labels)
                    
                    # Write metadata
                    for i, metadata in enumerate(batch_metadata):
                        record_group = metadata_group.create_group(str(current_idx + i))
                        for key, value in metadata.items():
                            record_group.attrs[key] = str(value)
                    
                    current_idx += batch_size
                    
                    # Clear batches
                    batch_images = []
                    batch_labels = []
                    batch_metadata = []
                    
                    # Flush to disk
                    f.flush()
                
            except Exception as e:
                logger.error(f"Error processing patch {patch_id}: {str(e)}")
                continue
        
        pbar.close()
    
    logger.info(f"Successfully saved {num_samples} samples to {cache_file}")
    logger.info("Preprocessing complete!")

def main():
    parser = argparse.ArgumentParser(description='Preprocess BigEarthNet dataset and create cached files')
    parser.add_argument('--image-dir', required=True, help='Directory containing BigEarthNet images')
    parser.add_argument('--metadata', required=True, help='Path to metadata parquet file')
    parser.add_argument('--output-dir', required=True, help='Directory to save cached files')
    parser.add_argument('--batch-size', type=int, default=3000, help='Batch size for processing')
    parser.add_argument('--splits', nargs='+', default=['test', 'validation', 'train'],
                      help='Dataset splits to process (default: train validation test)')
    parser.add_argument('--log-file', help='Path to log file (optional)')
    
    args = parser.parse_args()
    
    # Set up file logging if specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    for split in args.splits:
        cache_file = os.path.join(args.output_dir, f'{split}.h5')
        if Path(cache_file).exists():
            logger.info(f"Skipping {split} split, file already exists")
            continue

        logger.info(f"Processing {split} split...")
        preprocess_dataset(
            image_directory=args.image_dir,
            metadata_file=args.metadata,
            cache_file=cache_file,
            split=split,
            batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()
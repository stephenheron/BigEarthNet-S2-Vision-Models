import rasterio
import numpy as np
from PIL import Image
import glob
import os

def create_true_color(directory_path, output_path=None):
   # Find bands
   b2 = glob.glob(os.path.join(directory_path, '*B02.tif'))[0]
   b3 = glob.glob(os.path.join(directory_path, '*B03.tif'))[0]
   b4 = glob.glob(os.path.join(directory_path, '*B04.tif'))[0]
   
   # Read bands
   with rasterio.open(b4) as red:
       R = red.read(1)
   with rasterio.open(b3) as green:
       G = green.read(1)
   with rasterio.open(b2) as blue:
       B = blue.read(1)

   # Stack and normalize
   rgb = np.dstack((R, G, B))
   rgb = rgb.astype(float)
   rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
   rgb = (rgb * 255).astype(np.uint8)

   # Convert to PIL Image and save
   if output_path:
       img = Image.fromarray(rgb)
       img.save(output_path)
   
   return rgb

# Usage 
directory = "/home/stephen/workspace/BigEarthNet-S2/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57/"
output_file = "true_color.png"  # or .jpg
rgb_image = create_true_color(directory, output_file)

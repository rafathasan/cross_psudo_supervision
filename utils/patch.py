from PIL import Image
from tqdm import tqdm
import os
import numpy as np
from utils.label import rgb_mask_to_encoded_mask

def create_and_save_patch(image_path, output_dir, patch_size=512, stride=256, ext='jpg', encode=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = Image.open(image_path)

    with tqdm() as pbar:
        # Loop over the image and create patches
        for i, y in enumerate(range(0, img.height - patch_size + 1, stride)):
            for j, x in enumerate(range(0, img.width - patch_size + 1, stride)):

                left = x
                top = y
                right = x+patch_size
                bot = y+patch_size

                if right > img.width:
                    x = right - img.width
                    right = img.width

                if bot > img.height:
                    y = bot - img.height
                    bot = img.height
                    
                # Crop the patch
                patch = img.crop((left, top, right, bot))

                if encode:
                    patch = Image.fromarray(rgb_mask_to_encoded_mask(np.array(patch)))
                
                # Save the patch to a file
                filename = f"{output_dir}/patch_{i:03d}_{j:03d}.{ext}"
                patch.save(filename)

                pbar.update(1)




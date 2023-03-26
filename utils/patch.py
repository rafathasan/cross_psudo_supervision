from PIL import Image
from tqdm import tqdm
import os
import numpy as np
from utils.label import rgb_mask_to_encoded_mask


def create_and_save_patch(image_path, output_dir, patch_size=512, stride=256, postfix="", encode=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = Image.open(image_path)

    if encode:
        assert len(np.unique(np.array(img))) <= 33, "encode enable, but mask has more classes than expected"

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
                # if np.sum(np.array(patch)) != 0:
                if encode:
                    patch_array = rgb_mask_to_encoded_mask(np.array(patch))
                    assert len(np.unique(patch_array)) <= 30, "patch has more classes then expected"


                # Save the patch to a file
                filename = f"{output_dir}/patch_{i:03d}_{j:03d}{postfix}.png"
                patch.save(filename, format='PNG', compress_level=0, optimize=False)

                pbar.update(1)

def create_image_patches(image_path, output_dir, patch_size=512, stride=256, patch_name_suffix=None, patch_function=None):
    """
    Takes an image file located at `image_path` and creates patches of size `patch_size` with a stride of `stride`.
    Applies the optional `patch_function` to each patch and saves the resulting patches in `output_dir`. The function
    also generates a label file in `output_dir` that contains the class label for each patch in total.

    Args:
        image_path (str): The path to the input image file.
        output_dir (str): The directory where the output patches and label file will be saved.
        patch_size (int , optional): The size of each patch. If an integer is provided, the patch is assumed
            to be square with sides of length `patch_size`.
            dimensions of the patch. Default is 512.
        stride (int , optional): The stride between consecutive patches. If an integer is provided, the stride
            is assumed to be the same in both dimensions.
            stride between consecutive patches. Default is 256.
        patch_name_suffix (str, optional): A string to append to the end of each patch's filename. Default is None.
        patch_function (function, optional): A function to apply to each patch before saving. The function should take
            a single argument (the patch) and return the modified patch. Default is None.

    Raises:

    Returns:
        None
    """
    assert os.path.isfile(image_path), f"Invalid image file: {image_path}"
    assert isinstance(patch_size, int) and patch_size > 0, "patch_size should be an integer."
    assert isinstance(stride, int) and stride > 0, "stride should be an integer."
    assert patch_function is None or callable(patch_function), "patch_function should be a callable function."
    

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not patch_name_suffix:
        patch_name_suffix = ""

    image = Image.open(image_path)

    assert len(image.mode) == 3, f"Invalid image mode {image.mode}"

    total = len(range(0, image.height - patch_size + 1, stride)) * len(range(0, image.width - patch_size + 1, stride))

    with tqdm(total= total) as pbar:
        # Loop over the image and create patches
        for i, y in enumerate(range(0, image.height - patch_size + 1, stride)):
            for j, x in enumerate(range(0, image.width - patch_size + 1, stride)):
                left = x
                top = y
                right = x+patch_size
                bot = y+patch_size

                if right > image.width:
                    x = right - image.width
                    right = image.width

                if bot > image.height:
                    y = bot - image.height
                    bot = image.height

                # Crop the patch
                patch = image.crop((left, top, right, bot))

                if patch_function:
                    patch = patch_function(patch)

                # Save the patch to a file
                
                filename = f"{output_dir}/patch_{i:03d}_{j:03d}{patch_name_suffix}.png"
                
                patch.save(filename, format='PNG', compress_level=0, optimize=False)

                pbar.update(1)
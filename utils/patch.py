from PIL import Image
from tqdm import tqdm
import os
import numpy as np

def create_image_patches(image_path, label_path, output_dir, patch_size=512, stride=256, patch_name_suffix=None, patch_function=None):
    """
    Takes an image file located at `image_path` and label file `label_path` and creates patches of size `patch_size` with a stride of `stride`.
    Applies the optional `patch_function` to label patch and saves the resulting patches in `output_dir`. Besides The function
    also generates a image patch in `output_dir`.

    Args:
        image_path (str): The path to the input image file.
        label_path (str): The path to the label image file.
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

    To-do:
    # - zeroing image by background label
    # - Remove full bg patches with image patch
    - add edge label augmentation
    """
    assert os.path.isfile(image_path), f"Invalid image file: {image_path}"
    assert os.path.isfile(label_path), f"Invalid label file: {label_path}"
    assert isinstance(patch_size, int) and patch_size > 0, "patch_size should be an integer."
    assert isinstance(stride, int) and stride > 0, "stride should be an integer."
    assert patch_function is None or callable(patch_function), "patch_function should be a callable function."
    

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not patch_name_suffix:
        patch_name_suffix = ""

    image = Image.open(image_path)

    label = Image.open(label_path)

    assert len(image.mode) == 3, f"Invalid image mode {image.mode}"
    assert len(label.mode) == 3, f"Invalid label mode {label.mode}"
    assert image.size == label.size, f"Both image and label have to be same size but image: {image.size} and label: {label.size}"

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
                image_patch = image.crop((left, top, right, bot))
                label_patch = label.crop((left, top, right, bot))

                image_patch_array = np.array(image_patch)
                label_patch_array = np.array(label_patch)

                # skip unnecessary background patch
                if np.sum(image_patch_array) == 0 or np.sum(label_patch_array) == 0:
                    continue

                # zeroing image by background label

                mask = (label_patch_array == [0, 0, 0]).all(axis=2)

                image_patch_array[mask] = [0, 0, 0]

                image_patch = Image.fromarray(image_patch_array)
                label_patch = Image.fromarray(label_patch_array)

                if patch_function:
                    label_patch = patch_function(label_patch)

                # Save the patch to a file
                
                image_filename = f"{output_dir}/patch_{i:03d}_{j:03d}.png"
                label_filename = f"{output_dir}/patch_{i:03d}_{j:03d}_gt.png"
                
                image_patch.save(image_filename, format='PNG', compress_level=0, optimize=False)
                label_patch.save(label_filename, format='PNG', compress_level=0, optimize=False)

                pbar.update(1)
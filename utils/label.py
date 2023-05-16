from PIL import Image
import numpy as np

# original 11 class map
# class_map = {(0, 0, 0):      0, # bg
#             (0, 255, 0):     1, # farmland
#             (0, 0, 255):     2, # water
#             (0, 255, 255):   3, # forest
#             (128, 0, 0):     4, # urban_structure
#             (255, 0, 255):   5, # rural_built_up
#             (255, 0, 0):     6, # urban_built_up
#             (160, 160, 164): 7, # road
#             (255, 255, 0):   8, # meadow
#             (255, 251, 240): 9, # marshland
#             (128, 0, 128):   10 # brick_factory
#             }

class_map = {(0, 0, 0):      0,  # bg
            (0, 255, 0):     1,  # farmland
            (0, 0, 255):     2,  # water
            (0, 255, 255):   3,  # forest
            (128, 0, 0):     4,  # urban_structure
            (255, 0, 255):   4,  # rural_built_up
            (255, 0, 0):     4,  # urban_built_up
            (160, 160, 164): 4,  # road
            (255, 255, 0):   5,  # meadow
            (255, 251, 240): 5,  # marshland
            (128, 0, 128):   4  # brick_factory
            }

def encode_bw_mask(rgb_mask, class_map=class_map):
    """
    Encode a PIL Image object containing a RGB mask into a binary mask with class labels.

    Parameters: 
    rgb_mask (PIL Image): An image object containing the RGB mask. 
    class_map (dict): A dictionary mapping RGB values to class labels. Keys should be tuples of length 3 and values should be integers. 

    Returns: 
    bw_mask (PIL Image): A binary mask with class labels. 
    """
    # Check if the object is a PIL Image object
    assert isinstance(rgb_mask, Image.Image), "Object is not a PIL Image"
    # Check that all keys are tuples of length 3
    assert all(isinstance(k, tuple) and len(k) == 3 for k in class_map.keys()), "Invalid keys in class_map"
    # Check that all values are integers
    assert all(isinstance(v, int) for v in class_map.values()), "Invalid values in class_map"

    rgb_mask_array = np.array(rgb_mask)
    num_classes = len(class_map)
    channels = 3

    assert rgb_mask_array.shape[2] == channels, f"mask should have 3 channels but found {rgb_mask_array.shape[2]}."
    assert len(np.unique(rgb_mask_array)) <= num_classes*channels, "rgb mask has more classes than expected"

    # Label encode the mask
    bw_mask = np.zeros((rgb_mask_array.shape[0], rgb_mask_array.shape[1]), dtype=np.uint8)
    for rgb_val, class_label in class_map.items():
        indices = np.where(np.all(rgb_mask_array == rgb_val, axis=-1))
        bw_mask[indices] = class_label

    assert len(bw_mask.shape) == 2, f"Invalid Shape {bw_mask.shape}"
    assert len(np.unique(bw_mask)) <= num_classes, f"Invalid number of classes {len(np.unique(bw_mask))}"

    bw_mask = Image.fromarray(bw_mask)

    return bw_mask

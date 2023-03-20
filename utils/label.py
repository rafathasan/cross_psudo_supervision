from PIL import Image
import numpy as np
from tqdm import tqdm

def shift_encoded_label(input_file_path, output_file_path):

    classes = ['background', 'farmland', 'water', 'forest',
                'urban_structure', 'rural_builtup', 'Uurban_builtup',
                'road', 'meadow', 'marshland', 'brick_factory']
                
    target_classes = ['background', 'farmland', 'water', 'forest',
                    'structure', 'meadow']
    class_map = {0: [0], 1:[1], 2:[2], 3:[3], 4:[4, 5, 6, 7, 10], 5:[8, 9]}

    img = np.array(Image.open(input_file_path).convert('L'))

    intensities_min = img.min()
    intensities_max = img.max()

    new_min = 0
    new_max = len(np.unique(img)) - 1

    intensities_shifted = np.interp(img, (intensities_min, intensities_max), (new_min, new_max)).round().astype(np.uint8)

    with tqdm(total= len(class_map) * 2) as pbar:

        for v in range(6):
            for k in class_map[v]:
                intensities_shifted[intensities_shifted == k] = v
                pbar.update(1)

        intensities_shifted = Image.fromarray(intensities_shifted)

        intensities_shifted.save(output_file_path)

def rgb_mask_to_encoded_mask(rgb_mask, class_map = None):
    assert rgb_mask.shape[2] == 3
    
    if class_map == None:
        ### original 11 class map
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

        ### class map reduced to 6 classes
        class_map = {(0, 0, 0):      0, # bg
                    (0, 255, 0):     1, # farmland
                    (0, 0, 255):     2, # water
                    (0, 255, 255):   3, # forest
                    (128, 0, 0):     4, # urban_structure
                    (255, 0, 255):   4, # rural_built_up
                    (255, 0, 0):     4, # urban_built_up
                    (160, 160, 164): 4, # road
                    (255, 255, 0):   5, # meadow
                    (255, 251, 240): 5, # marshland
                    (128, 0, 128):   4  # brick_factory
                    }

    # Label encode the mask
    encoded_mask = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1]), dtype=np.uint8)
    for rgb_val, class_label in class_map.items():
        indices = np.where(np.all(rgb_mask == rgb_val, axis=-1))
        encoded_mask[indices] = class_label

    return encoded_mask
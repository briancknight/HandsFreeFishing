import logging
import numpy as np
import figure_scripts
from segment_anything import SamPredictor, sam_model_registry
import pandas as pd
import os
import pickle
from HandsFreeFishing import fin_clipping 
from landmark_point_placement import *
        
# LOGGING SETUP
logging.basicConfig(filename='hff.log',
                    level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s'
)

model_path = os.path.join('..','sam_vit_l_0b3195.pth')

with open(os.path.join('..','weight_prediction_092025.pkl'), 'rb') as file:
    weight_model = pickle.load(file)

def run_adult_steelhead(im_paths, thaw=False, dir_name="LakeTaupo_example_fish"):

    # read in Meta's Segment Anything Model (SAM)
    sam = sam_model_registry['vit_l'](checkpoint=model_path)
    predictor = SamPredictor(sam) 
    
    # options optional fins to clip include: ['dorsal', 'adipose', 'caudal', 'anal', 'pelvic', 'pectoral']
    fins_to_clip = ['dorsal', 'adipose', 'caudal', 'anal', 'pelvic', 'pectoral']
    
    landmark_img_dir=os.path.join('landmark_point_images', dir_name)
    os.makedirs(landmark_img_dir,exist_ok=True)
    
    landmark_dir=os.path.join(f'landmark_point_data',dir_name)
    os.makedirs(landmark_dir,exist_ok=True)
    
    save_segmentations=True
    
    for (i,im_path) in enumerate(im_paths):
        print(im_path)
        fish = fin_clipping.adult_stealhead(im_path, predictor,write_masks=True,fins_to_clip=fins_to_clip, ord=100)
        fish.check_freezer()
        if not fish.frozen:
            
            try:
                fish.run()
                if fish.can_run:
                    if save_segmentations:
                        fish.write_full_masks()
                        fish.write_fin_masks()
                
                    # from the fish object, access morphometric features to be saved to a spreadsheet
                    # qualities.append(fish.quality)
                    
                    # Fork length, scale, and surface areas
                    # FLs.append(fish.FL)
                    # scales.append(fish.scale)
                    # areas.append(fish.area)
                    # no_fin_areas.append(fish.no_fin_area)
                        
                    landmark_points = get_landmark_points(fish)
                    
                    save_landmark_image(fish, landmark_points, name=os.path.join(landmark_img_dir,fish.im_name))
                    
                    np.save(os.path.join(landmark_dir, fish.im_name+'_landmark_points.npy'), np.array(landmark_points))
                
            except Exception as e:
                logging.exception(f"An unexpected error occurred while processing {im_path}; please see error log to diagnose the isssue further")
                

    

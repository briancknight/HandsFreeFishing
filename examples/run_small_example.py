import numpy as np
import figure_scripts
from segment_anything import SamPredictor, sam_model_registry
import pandas as pd
import os
from HandsFreeFishing import fin_clipping 
import pickle
        
model_path = os.path.join('..','sam_vit_l_0b3195.pth') # please be sure to place the SAM model in the HandsFreeFishing Directory

with open(os.path.join('..','weight_prediction_092025.pkl'), 'rb') as file:
    weight_model = pickle.load(file)
 
def main():
    dir_name = "example_fish"
    full_dir = os.path.join("sushi", dir_name)
    excel_dir = 'spreadsheets'
    df = pd.read_excel(os.path.join(excel_dir, dir_name+'.xlsx'),nrows=4,sheet_name=0)
    
    im_names = df["FIshID"]
    im_paths = [os.path.join(full_dir, id + ".jpg") for id in im_names]
    
    # read in Meta's Segment Anything Model (SAM)
    print('reading sam...')
    sam = sam_model_registry['vit_l'](checkpoint=model_path)
    print('sam read')
    predictor = SamPredictor(sam) 
    
    qualities=[]
    
    # fork length, image scale, area, and no fin area
    FLs=[]
    scales=[]
    areas=[]
    no_fin_areas=[]
    
    # major any minor axis of an ellise of best fit for the fish contour
    major_axes = []
    minor_axes = []
    
    # other morphometric measurements
    eye_diameters=[]
    partitioned_areas=[]
    partition_line_lengths=[]
    landmark_lengths=[]
    
    # options
    save_xlsx = True
    save_segmentations = True
    predict_weights = True
    
    for (i,im_path) in enumerate(im_paths[:4]):
        
        fish = fin_clipping.fish(im_path, predictor,write_masks=True)
        fish.run()
        
        if save_segmentations:
            fish.write_full_masks()
            fish.write_fin_masks()
        
        # from the fish object, access morphometric features to be saved to a spreadsheet
        qualities.append(fish.quality)
        
        # Fork length, scale, and surface areas
        FLs.append(fish.FL)
        scales.append(fish.scale)
        areas.append(fish.area)
        no_fin_areas.append(fish.no_fin_area)
        
        # other morphometric features
        eye_diameters.append(fish.eye_diameter)
        major_axes.append(fish.major_axis)
        minor_axes.append(fish.minor_axis)
        partitioned_areas.append(fish.partitioned_areas)
        partition_line_lengths.append(fish.partition_line_lengths)
        
        # landmark lengths
        fish.get_digitized_landmarks(n_steps=7)
        landmark_lengths.append(np.array(fish.best_truss_lengths)*fish.scale)
        
        if im_path==im_paths[3]:
            figure_scripts.save_example_figures(fish, im_names[i])
            figure_scripts.save_truss_figure(fish, im_names[i])
    
    # weight prediction
    if predict_weights:
        explanatory_var = np.array(no_fin_areas) * np.array(minor_axes)
        predicted_weights = weight_model.predict(explanatory_var.reshape(-1,1))
        
    if save_xlsx:
        
        if predict_weights:
            df["pred weights"]=predicted_weights
            
        df["quality"] = qualities
        df["pred scale"] = scales
        df["pred FL"] = FLs
        df["pred area"] = areas
        df["pred area (no fins)"] = no_fin_areas
        df["eye_diameters"]=eye_diameters
        df["major axis"]=major_axes
        df["minor axis"]=minor_axes

        for i in range(len(fish.partitioned_areas)):
            df["Partition Area " + str(i)] = np.array(partitioned_areas)[:,i]
        for i in range(len(fish.partition_line_lengths)):
            df["Sector Length " + str(i)] = np.array(partition_line_lengths)[:,i]
        
        # first convert to a np array for slicing
        landmark_lengths = np.array(landmark_lengths)
        for i in range(len(landmark_lengths[0])):
            df["Landmark Length " + str(i)] = landmark_lengths[:, i]
        df.to_excel(os.path.join('measurements', dir_name,'output_small.xlsx'))
    
    
if __name__ == "__main__":
    main()


    

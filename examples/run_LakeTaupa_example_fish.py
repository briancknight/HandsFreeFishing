import numpy as np
import figure_scripts
from segment_anything import SamPredictor, sam_model_registry
import pandas as pd
import os
import pickle
from HandsFreeFishing import fin_clipping 
from landmark_point_placement import *
        
     
model_path = os.path.join('..','sam_vit_l_0b3195.pth')

with open(os.path.join('..','weight_prediction_092025.pkl'), 'rb') as file:
    weight_model = pickle.load(file)
    
def main(dir_name="subset", spreadsheet_name="subset"):
    raw_data_dir = os.path.join("sushi", dir_name)
    excel_dir = 'spreadsheets'
    df = pd.DataFrame()#pd.read_excel(os.path.join(excel_dir, spreadsheet_name+'.xlsx'),sheet_name=0)
    
    if dir_name=="LakeTaupo_example_fish":
        im_names=[f"{i}_2024" for i in [5,6]]
        im_paths = [os.path.join(raw_data_dir, id + ".jpg") for id in im_names]
    
    if dir_name=="subset":
        df=pd.read_csv(os.path.join('spreadsheets', spreadsheet_name+'.csv'))
        im_names = [f"{str(df['ID'][i])}_{df['overalldiet'][i]}" for i in range(df.shape[0])]
        im_paths = [os.path.join(raw_data_dir,im_name+'.jpeg') for im_name in im_names]
    # read in Meta's Segment Anything Model (SAM)
    sam = sam_model_registry['vit_l'](checkpoint=model_path)
    predictor = SamPredictor(sam) 
    
    # image quality as judged by user
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
    
    # options optional fins to clip include: ['dorsal', 'adipose', 'caudal', 'anal', 'pelvic', 'pectoral']
    fins_to_clip = ['dorsal', 'adipose', 'caudal', 'anal', 'pelvic']
    save_xlsx = True
    save_segmentations = True
    predict_weights = True
    
    landmark_img_dir=os.path.join('landmark_point_images', dir_name)
    os.makedirs(landmark_img_dir,exist_ok=True)
    
    landmark_dir=os.path.join(f'landmark_point_data',dir_name)
    os.makedirs(landmark_dir,exist_ok=True)
        
    for (i,im_path) in enumerate(im_paths):
        
        fish = fin_clipping.adult_stealhead(im_path, predictor,write_masks=True,fins_to_clip=fins_to_clip, ord=100)
        fish.run()
        # fish.get_full_segmentations()
        
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
        
        if im_path in im_paths:
            print('creating FL and truss figures...')
            figure_scripts.save_example_figures(fish, im_names[i])
            # figure_scripts.save_truss_figure(fish, im_names[i])
            
        landmark_points = get_landmark_points(fish)
        
        save_landmark_image(fish, landmark_points, name=os.path.join(landmark_img_dir,fish.im_name))
        
        np.save(os.path.join(landmark_dir, fish.im_name+'_landmark_points.npy'), np.array(landmark_points))
    
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
        df.to_excel(os.path.join('measurements', dir_name,'output.xlsx'))

if __name__ == "__main__":
    main(dir_name="LakeTaupo_example_fish")


    

import os
import cv2 as cv
import numpy as np
import pandas as pd
from HandsFreeFishing import postprocess_landmark_points
    
# dir_name="subset"
# spreadsheet_name="subset"

dir_name="LakeTaupo_example_fish"
spreadsheet_name=''

raw_data_dir = os.path.join("sushi", dir_name)
landmark_data_dir = os.path.join("landmark_point_data", dir_name)
excel_dir = 'spreadsheets'
df = pd.DataFrame()#pd.read_excel(os.path.join(excel_dir, spreadsheet_name+'.xlsx'),sheet_name=0)

if dir_name=="LakeTaupo_example_fish": # define image names directly
    im_names=[os.path.join(f"{i}_2024") for i in [5,6]]
    im_paths = [os.path.join(raw_data_dir, id + ".jpg") for id in im_names]
    landmark_point_paths = [os.path.join(landmark_data_dir,im_name+'_landmark_points.npy') for im_name in im_names]

if dir_name=="subset": # read in data from spreadsheet
    df=pd.read_csv(os.path.join('spreadsheets', spreadsheet_name+'.csv'))
    im_names = [f"{str(df['ID'][i])}_{df['overalldiet'][i]}" for i in range(df.shape[0])]
    im_paths = [os.path.join(raw_data_dir,im_name+'.jpeg') for im_name in im_names]
    landmark_point_paths = [os.path.join(landmark_data_dir,im_name+'_landmark_points.npy') for im_name in im_names]

postprocess_landmark_points(im_paths, dir_name,landmark_data_dir,name_change='_updated', monitor_idx=0)
""" if name_change='', running the postprocessing will overwrite previously computed landmark points with adjusted points,
and similarly with the landmark point images

monitor_idx should only be changed if a multiple monitor display is buing used, and should be adjusted by the user to place pop-up
windows on the desired monitor
"""
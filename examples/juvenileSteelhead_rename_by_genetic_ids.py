import argparse
import pandas as pd
import numpy as np
import os
import glob
import json

def get_paths_to_delete(project_name, im_name):
    
    paths=[]
    paths.append(os.path.join("measurements", project_name, im_name+'_fin_points.npy'))
    paths.append(os.path.join("measurements", project_name, im_name+'.csv'))
    for fin in ['dorsal','adipose', 'caudal','anal','pelvic','pectoral','eye','initial','full']:
        paths.append(os.path.join("segmentations", project_name, fin+'_masks', fin+'_mask_' + im_name+'.png'))
    paths.append(os.path.join("landmark_point_data", project_name, im_name+'_landmark_points.npy'))
    paths.append(os.path.join("landmark_point_images", project_name, im_name+'_landmark_points.png'))
    
    return paths

parser = argparse.ArgumentParser("HandsFreeFishing project")
# parser.add_argument('--project_name', type=str, default="LakeTaupo_example_fish", help='Name of project directory')
parser.add_argument('-c', '--convert', type=int, default=0, help="0 convert from image id to genetic id, 1 does the opposite")

args = parser.parse_args()


# Example lines:
# python lakeTaupo_project -s 5_2024 -p -r -o (equivalent to: python lakeTaupo_project --single-fish 5_2024 --preprocess --run --postprocess)
# python lakeTaupo_project -d 5_2024 

if __name__ == "__main__":
    
    # DEFINE PROJECT DETAILS HERE, INCLUDING DATA FRAME
    
    # NAME OF THE DATA FRAME/EXCEL SPREADSHEET
    spreadsheet_name="TDC_all_genetic_ID_2026"
    project_name="juvenileSteelhead_FEH5"
    if args.convert==0:
        rename_by_genetic_id=True# TO RENAME ALL RECOGNIZED IMAGES BY GENETIC ID
        rename_by_photo_id=False # TO RENAME ALL RECOGNIZED IMAGES BY IMAGE ID (undo previous process)
        
    else:
        rename_by_genetic_id=False# TO RENAME ALL RECOGNIZED IMAGES BY GENETIC ID
        rename_by_photo_id=True # TO RENAME ALL RECOGNIZED IMAGES BY IMAGE ID (undo previous process) 
        
    # END OF PROJECT DETAILS TO DEFINE
    
    spreadsheet_dir_name=os.path.join("spreadsheets", spreadsheet_name+".xlsx")
    df = pd.read_excel(spreadsheet_dir_name, sheet_name=1)
    
    def get_id_no(image_id_string,split_string='-'):
        if isinstance(image_id_string,float):
            return np.nan
        else:
            split_id = image_id_string.split(split_string)
            if len(split_id)<2:
                return np.nan
            else:
                return split_id[1]
            
    image_ids = df['image_id']
    image_id_nos = [get_id_no(id) for id in image_ids]
        
    dir_name=os.path.join("sushi", project_name)

    im_path_exts= glob.glob(os.path.join(dir_name,'*')) + glob.glob(os.path.join(dir_name,'*/*'))
    # im_path_exts=['sushi/juvenileSteelhead_FEH5/juvenileSteelhead_FEH5T/FEH5T_2026.03.13_01.JPG']
                  #'sushi/juvenileSteelhead_FEH5/juvenileSteelhead_FEH5T/FEH5T_2026.03.13_01.JPG']
    
    if rename_by_genetic_id:
        for im_path in im_path_exts:
            dir,im_name_ext=os.path.split(im_path)
            im_name, ext = os.path.splitext(im_name_ext)
            # print(f'dir={dir}, im_name_ext = {im_name_ext}, im_name={im_name},ext={ext}')
            im_id = get_id_no(im_name,split_string='_')
            try: 
                idx=image_id_nos.index(im_id)
                genetic_id = df['genetic_id'][idx]
                
                old_image_name=im_path
                new_image_name=os.path.join(dir, genetic_id+ext)
                print(f'old name = {old_image_name}')
                print(f'new name = {new_image_name}')
                os.replace(old_image_name, new_image_name)
                old_paths=get_paths_to_delete(project_name, im_name)
                new_paths=get_paths_to_delete(project_name, genetic_id)
                for (i,old_path) in enumerate(old_paths):
                    if os.path.exists(old_path):
                        os.replace(old_path, new_paths[i])
                    
                # print(f"data found for {im_path} with genetic id {genetic_ids[-1]}")
                # print(f"life stage is: {life_stages[-1]}\n")
                # print(f"genetic id is: {genetic_ids[-1]}\n")
            except ValueError:
                print(f'\nno data found for im_path={im_path}\n')
                pass
        
    if rename_by_photo_id:
        genetic_ids=list(df['genetic_id'])
        im_ids=np.array(df['image_id'])
        for im_path in im_path_exts:
            print(im_path)
            dir,im_name_ext=os.path.split(im_path)
            genetic_id, ext = os.path.splitext(im_name_ext)
            try: 
                idx=genetic_ids.index(genetic_id)
                im_id = im_ids[idx]
                im_id_no = get_id_no(im_id,split_string='-')
                old_image_name=im_path
                new_image_name=os.path.join(dir, f'IMG_{im_id_no}{ext}')
                print(f'old name = {old_image_name}')
                print(f'new name = {new_image_name}')
                os.replace(old_image_name, new_image_name)
                old_paths=get_paths_to_delete(project_name, genetic_id)
                new_paths=get_paths_to_delete(project_name, f'IMG_{im_id_no}')
                for (i,old_path) in enumerate(old_paths):
                    if os.path.exists(old_path):
                        os.replace(old_path, new_paths[i])
                    
                # print(f"data found for {im_path} with genetic id {genetic_ids[-1]}")
                # print(f"life stage is: {life_stages[-1]}\n")
                # print(f"genetic id is: {genetic_ids[-1]}\n")
            except ValueError:
                print(f'\nno data found for im_path={im_path}\n')
                pass
    
            
    

            
  
        
    
        
        
                
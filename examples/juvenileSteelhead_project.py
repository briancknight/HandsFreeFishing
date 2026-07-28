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
parser.add_argument('-s', '--single-fish', type=str, default=None, help="specify a single image name to run")
parser.add_argument('-c', '--cont', type=str, default=None, help="continue processing at a given image name")
parser.add_argument('-d', '--delete', type=str, default=None, help='delete all data corresponding to a given image name')
parser.add_argument('-p', '--preprocess', default = False, action=argparse.BooleanOptionalAction, help='Initiate preprocessing (default: True)')
parser.add_argument('-r', '--run', default=False, action=argparse.BooleanOptionalAction, help='Initiate automated segmentation (default: True)')
parser.add_argument('-o', '--postprocess', default=False, action=argparse.BooleanOptionalAction, help='Initiate postprocessing default: True)')
# parser.add_argument('-t', '--tps', default=False, action=argparse.BooleanOptionalAction, help='Create tps file from landmark point npy files: True)')
parser.add_argument('-m', '--monitor', type=int, default=0, help='Index of monitor to be used for image displays')

args = parser.parse_args()


# Example lines:
# python lakeTaupo_project -s 5_2024 -p -r -o (equivalent to: python lakeTaupo_project --single-fish 5_2024 --preprocess --run --postprocess)
# python lakeTaupo_project -d 5_2024 

if __name__ == "__main__":
    
    # DEFINE PROJECT DETAILS HERE, INCLUDING DATA FRAME
    
    # NAME OF THE DATA FRAME/EXCEL SPREADSHEET
    spreadsheet_name="TDC_all_genetic_ID_2026"
    project_name="juvenileSteelhead_FEH5"
    using_genetic_ids=True
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

    im_path_exts= glob.glob(os.path.join(dir_name,'*')) + glob.glob(os.path.join(dir_name,'*','*'))
    
    if using_genetic_ids:
        im_paths=[]
        im_names=[]
        genetic_ids_found=[]
        exts=[]
        genetic_ids=list(df['genetic_id'])
        life_stages=[]
        im_dataframe_idxs=[]
        
        for im_path in im_path_exts:
            dir,im_name_ext=os.path.split(im_path)
            genetic_id, ext = os.path.splitext(im_name_ext)
            try: 
                idx=genetic_ids.index(genetic_id)
                genetic_ids_found.append(genetic_id)
                im_dataframe_idxs.append(idx)
                im_paths.append(os.path.join(dir, genetic_id))
                im_names.append(genetic_id)
                exts.append(ext)
                life_stages.append(df['life_stage'][idx])
            except ValueError:
                print(f'\nno data found for im_path={im_path}\n')
                pass
    else:
        im_paths=[]
        im_names=[]
        exts=[]
        genetic_ids=[]
        life_stages=[]
        im_dataframe_idxs=[]
        
        for im_path in im_path_exts:
            dir,im_name_ext=os.path.split(im_path)
            im_name, ext = os.path.splitext(im_name_ext)
            # print(f'dir={dir}, im_name_ext = {im_name_ext}, im_name={im_name},ext={ext}')
            im_id = get_id_no(im_name,split_string='_')
            try: 
                idx=image_id_nos.index(im_id)
                im_dataframe_idxs.append(idx)
                im_paths.append(os.path.join(dir, im_name))
                im_names.append(im_name)
                exts.append(ext)
                genetic_ids.append(df['genetic_id'][idx])
                life_stages.append(df['life_stage'][idx])
                # print(f"data found for {im_path} with genetic id {genetic_ids[-1]}")
                # print(f"life stage is: {life_stages[-1]}\n")
                # print(f"genetic id is: {genetic_ids[-1]}\n")
            except ValueError:
                print(f'\nno data found for im_path={im_path}\n')
                pass
    
    if args.delete is not None:
        paths=get_paths_to_delete(project_name, args.delete)
        
        confirmation = input(f"Are you sure you want to proceed with this deleting data relevent to {args.delete}? (y/n): ").strip().lower()
        if confirmation=='y':
            for path in paths:
                if os.path.exists(path):
                    os.remove(path)
            print('all data deleted')
        else:
            print('action cancelled')
            
    if args.single_fish is not None:
        # overwrite to process the specific image
        if using_genetic_ids:
            # print(genetic_ids)
            idx = genetic_ids_found.index(args.single_fish)
            im_names = im_names[idx:idx+1]
            im_paths = im_paths[idx:idx+1]
            print(im_paths)
            im_path_exts = im_path_exts[idx:idx+1]
            genetic_ids = genetic_ids[idx:idx+1]
            life_stages = [df['life_stage'][idx]]
        else:
            idx = im_names.index(args.single_fish)
            im_names = im_names[idx:idx+1]
            im_paths = im_paths[idx:idx+1]
            print(im_paths)
            im_path_exts = im_path_exts[idx:idx+1]
            
            im_id = get_id_no(im_names[0],split_string='_')
            idx2=image_id_nos.index(im_id)
            genetic_ids = [df['genetic_id'][idx2]]
            life_stages = [df['life_stage'][idx2]]
        
    if args.cont is not None:
        # continue processing starting with a given fish id
        idx = im_names.index(args.cont)
        im_names=im_names[idx:]
        im_paths=im_paths[idx:]
            
    if args.preprocess:
        from HandsFreeFishing import preprocess_juvenile_steelhead
        """
        landmark_length: sets the expected length between the reference points to compute the image scale
        
        ds: stands for down-sample, set to either, 1, 2, 4, or 8. ds=1 is full resolution, ds=8 downsamples by a factor of 8, 
        and will result in lower resolution image displays which can potentiall speed up processing
        
        monitor_idx should only be changed if a multiple monitor display is buing used, and should be adjusted by the user to place pop-up
        windows on the desired monitor
        """
        preprocess_juvenile_steelhead(im_paths, exts, genetic_ids, life_stages, project_dir=project_name, landmark_length=50,ds=1,monitor_idx=args.monitor)
        
    if args.run:
        from run_scripts import run_juvenile_steelhead
        run_juvenile_steelhead(im_paths, life_stages, genetic_ids, dir_name=project_name)
        
    if args.postprocess:
        import openpyxl
        
        data_file_name = os.path.join('measurements', project_name, 'predicted_measurements.txt')
        
        with open(data_file_name, "r",encoding="utf-8") as file:
            data=json.load(file)
        
        predicted_data_df = pd.DataFrame(data)
        
        insert_col_idx= 1+df.columns.get_loc('fl_mm')
        
        for new_column in ['predicted_fl_mm','yolk_sac_SA_mm^2', 'SA_mm^2','no_fin_no_yolk_sac_SA_mm^2']:
            df.insert(insert_col_idx, new_column, value=np.nan)
            insert_col_idx+=1
            temp_column = df[new_column].copy()
            for (i,im_name) in enumerate(im_names):
                df_idx=im_dataframe_idxs[i]                
                temp_column[df_idx] = predicted_data_df[im_name][new_column]
                
            df[new_column] = temp_column
                
        os.makedirs(os.path.join('spreadsheets',project_name),exist_ok=True)
        df.to_excel(os.path.join('spreadsheets',project_name,'updated_data.xlsx'))
        
    
        
        
                
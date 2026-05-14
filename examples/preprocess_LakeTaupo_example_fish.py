import pandas as pd
import os
from HandsFreeFishing import preprocess_adult_steelhead_updated
    
def pre_process(project_name="LakeTaupo_example_fish", spreadsheet_name="LakeTaupo_example_fish"):
    dir_name = os.path.join("sushi",project_name)
    # dir = "spreadsheets"
    # df = pd.read_excel(os.path.join(dir,spreadsheet_name+".xlsx"), sheet_name=0)
    if project_name=="subset":
        df=pd.read_csv(os.path.join('spreadsheets', spreadsheet_name+'.csv'))
        im_paths = [os.path.join(dir_name, f"{str(df['ID'][i])}_{df['overalldiet'][i]}.jpeg") for i in range(df.shape[0])]
    else:
        im_names=[os.path.join(f"{i}_2024") for i in [5,6]]
        im_paths = [os.path.join(dir_name, id + ".jpg") for id in im_names]
        
    preprocess_adult_steelhead_updated(im_paths,landmark_length=50,ds=1,monitor_idx=1)
    """
    landmark_length: sets the expected length between the reference points to compute the image scale
    
    ds: stands for down-sample, set to either, 1, 2, 4, or 8. ds=1 is full resolution, ds=8 downsamples by a factor of 8, 
    and will result in lower resolution image displays which can potentiall speed up processing
    
    monitor_idx should only be changed if a multiple monitor display is buing used, and should be adjusted by the user to place pop-up
    windows on the desired monitor
    """
    
if __name__ == "__main__":
    pre_process(project_name="LakeTaupo_example_fish", spreadsheet_name="subset")
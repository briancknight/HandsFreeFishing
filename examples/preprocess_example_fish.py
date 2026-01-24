import pandas as pd
import os
from HandsFreeFishing import get_rois_flips_and_bad_paths
    
def pre_process(project_name="example_fish", spreadsheet_name="example_fish"):
    dir_name = os.path.join("sushi",project_name)
    dir = "spreadsheets"
    df = pd.read_excel(os.path.join(dir,spreadsheet_name+".xlsx"), sheet_name=0)
    
    im_names = df["FishID"]
    im_paths = [os.path.join(dir_name, id + ".jpg") for id in im_names]
    # print(im_paths) uncomment for debugging
    get_rois_flips_and_bad_paths(im_paths)
    
if __name__ == "__main__":
    # main()
    pre_process(project_name="example_fish", spreadsheet_name="example_fish")
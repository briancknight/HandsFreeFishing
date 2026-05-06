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
        im_names=[os.path.join(f"{i}_2024") for i in range(5,22)]
        im_paths = [os.path.join(dir_name, id + ".jpg") for id in im_names]
        
    print(im_paths)
    # print(im_paths) uncomment for debugging
    preprocess_adult_steelhead_updated(im_paths,landmark_length=50)
    
if __name__ == "__main__":
    # main()
    # pre_process(project_name="LakeTaupo_example_fish", spreadsheet_name="LakeTaupo_example_fish")
    pre_process(project_name="LakeTaupo_example_fish", spreadsheet_name="subset")
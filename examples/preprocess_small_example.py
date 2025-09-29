import pandas as pd
import os
from HandsFreeFishing import get_rois_flips_and_bad_paths


def main():
    dir_name = os.path.join("sushi","example_fish")
    dir = "spreadsheets"
    df = pd.read_excel(os.path.join(dir,'example_fish.xlsx'), sheet_name=0)
    
    im_names = df["FIshID"]
    im_paths = [os.path.join(dir_name, id + ".jpg") for id in im_names]
    get_rois_flips_and_bad_paths(im_paths[:4])
    
if __name__ == "__main__":
    main()
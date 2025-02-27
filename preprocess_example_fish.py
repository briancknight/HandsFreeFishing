import pandas as pd
import os
import preprocessing


def main():
    dir_name = "/Users/brknight/Documents/GitHub/HandsFreeFishing/sushi/example_fish/"
    dir = '/Users/brknight/Documents/GitHub/HandsFreeFishing/Spreadsheets/'
    df = pd.read_excel(os.path.join(dir,'example_fish.xlsx'),sheet_name=0)
    
    im_names = df["FIshID"]
    im_paths = [dir_name + id + ".jpg" for id in im_names]
    print(im_paths)
    preprocessing.get_rois_flips_and_bad_paths(im_paths)
    
if __name__ == "__main__":
    main()
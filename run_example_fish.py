import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import pandas as pd
import os
import fin_clipping 
        
        
def main():
    dir_name = "example_fish"
    full_dir = os.path.join("sushi", dir_name)
    excel_dir = 'Spreadsheets'
    df = pd.read_excel(os.path.join(excel_dir, dir_name+'.xlsx'),sheet_name=0)
    
    im_names = df["FIshID"]
    im_paths = [os.path.join(full_dir, id + ".jpg") for id in im_names]
    
    # read in Meta's Segment Anything Model (SAM)
    sam = sam_model_registry['vit_l'](checkpoint="sam_vit_l_0b3195.pth")
    predictor = SamPredictor(sam) 
    school=[]
    areas=[]
    no_fin_areas=[]
    sector_areas=[]
    FLs=[]
    scales=[]
    qualities=[]
    save_xlsx = True
    
    for im_path in im_paths:
        
        fish = fin_clipping.fish(im_path, predictor,write_masks=True)
        fish.run()
        school.append(fish)
        fish.write_full_masks()
        fish.write_fin_masks()
        fish.filet_fish()
        FLs.append(fish.FL)
        scales.append(fish.scale)
        areas.append(fish.area)
        qualities.append(fish.quality)
        no_fin_areas.append(fish.no_fin_area)
        sector_areas.append(fish.sector_areas)
        
    
    if save_xlsx:
        df["quality"] = qualities
        df["pred scale"] = scales
        df["pred FL"] = FLs
        df["pred area"] = areas
        df["pred area (no fins)"] = no_fin_areas
        for i in range(len(fish.sector_areas)):
            df["Sector Area " + str(i)] = np.array(sector_areas)[:,i]
        df.to_excel(os.path.join('measurements', dir_name,'output.xlsx'))
    
    
if __name__ == "__main__":
    main()


    

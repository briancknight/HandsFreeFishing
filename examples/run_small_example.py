import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import pandas as pd
import os
from HandsFreeFishing import fin_clipping 
        
        
model_path = os.path.join('..','sam_vit_l_0b3195.pth')

def main():
    dir_name = "example_fish"
    full_dir = os.path.join("sushi", dir_name)
    excel_dir = 'spreadsheets'
    df = pd.read_excel(os.path.join(excel_dir, dir_name+'.xlsx'),nrows=4,sheet_name=0)
    
    im_names = df["FIshID"]
    im_paths = [os.path.join(full_dir, id + ".jpg") for id in im_names]
    
    # read in Meta's Segment Anything Model (SAM)
    sam = sam_model_registry['vit_l'](checkpoint=model_path)
    predictor = SamPredictor(sam) 
    school=[]
    areas=[]
    no_fin_areas=[]
    sector_areas=[]
    line_lengths=[]
    eye_diameters=[]
    FLs=[]
    scales=[]
    qualities=[]
    save_xlsx = True
    
    for im_path in im_paths[:4]:
        
        fish = fin_clipping.fish(im_path, predictor,write_masks=True)
        fish.run()
        school.append(fish)
        fish.write_full_masks()
        fish.write_fin_masks()
        # fish.filet_fish()
        FLs.append(fish.FL)
        scales.append(fish.scale)
        areas.append(fish.area)
        qualities.append(fish.quality)
        no_fin_areas.append(fish.no_fin_area)
        sector_areas.append(fish.sector_areas)
        line_lengths.append(fish.line_lengths)
        eye_diameters.append(fish.eye_diameter)
        
    
    if save_xlsx:
        df["quality"] = qualities
        df["pred scale"] = scales
        df["pred FL"] = FLs
        df["pred area"] = areas
        df["pred area (no fins)"] = no_fin_areas
        df["eye_diameters"]=eye_diameters
        for i in range(len(fish.sector_areas)):
            df["Sector Area " + str(i)] = np.array(sector_areas)[:,i]
            df["Sector Length " + str(i)] = np.array(line_lengths)[:,i]  
            
        df.to_excel(os.path.join('measurements', dir_name,'output_small.xlsx'))
    
    
if __name__ == "__main__":
    main()


    

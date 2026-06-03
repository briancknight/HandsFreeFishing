from py_tps import TPSFile, TPSImage, TPSCurve, TPSPoints
import numpy as np
import pandas as pd
import os
import glob
import csv 
   

def generate_tps_file(im_paths, name_change='',landmark_data_dir='landmark_point_data', measurement_data_dir='measurements'):
    
    project_name= os.path.split(os.path.split(im_paths[0])[0])[1]
    im_names=[os.path.split(im_path)[1] for im_path in im_paths]
    landmark_point_paths = [os.path.join(landmark_data_dir, project_name, im_name+'_landmark_points.npy') for im_name in im_names]
    measurement_paths = [os.path.join(measurement_data_dir, project_name, im_name+'.csv') for im_name in im_names]
    os.makedirs('TPS_data', exist_ok=True)
    tps_images=[]
    
    for (i, point_path) in enumerate(landmark_point_paths):
        
        found_files = glob.glob(f"{im_paths[i]}.*")
        if len(found_files)>=1:
            ext = os.path.splitext(found_files[0])[1]
            # 1. Define landmarks (x, y coordinates)
            if os.path.exists(point_path):
                points = TPSPoints(np.load(point_path))
                # 2. Create a curve from those points
                curve = TPSCurve(points)

                with open(measurement_paths[i], newline='') as csvfile:
                            reader=csv.reader(csvfile, delimiter=',')
                            for (j,row) in enumerate(reader):
                                if j==0:
                                    scale = row[0]
                # 3. Create an image entry with landmarks, curves, and metadata
                image = TPSImage(im_names[i]+ext, 
                    landmarks=points, 
                    id_number=i, 
                    comment=f"comment for fish {im_names[i]}", 
                    scale=scale
                )

                tps_images.append(image)
    # 4. Construct the TPS object
    tps_file = TPSFile(tps_images)

    # 5. Write to a file
    tps_file.write_to_file(os.path.join('TPS_data',f'{project_name}{name_change}.TPS'))


def main():

    project_name="LakeTaupo_example_fish"
    raw_data_dir = os.path.join("sushi", project_name)
    im_names = ['5_2024', '6_2024']
    im_paths = [os.path.join(raw_data_dir, im_name) for im_name in im_names]
    generate_tps_file(im_paths, name_change='', landmark_data_dir='landmark_point_data')
    
if __name__=='__main__':
    main()
    
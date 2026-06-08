import argparse
import pandas as pd
import os

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
parser.add_argument('-t', '--tps', default=False, action=argparse.BooleanOptionalAction, help='Create tps file from landmark point npy files: True)')
parser.add_argument('-m', '--monitor', type=int, default=0, help='Index of monitor to be used for image displays')
parser.add_argument('-nc', '--name-change', type=str, default='', help='name for updated landmarks after postprocessing')

args = parser.parse_args()


# Example lines:
# python lakeTaupo_project -s 5_2024 -p -r -o (equivalent to: python lakeTaupo_project --single-fish 5_2024 --preprocess --run --postprocess)
# python lakeTaupo_project -d 5_2024 

if __name__ == "__main__":
    
    # define project details here, including dataframe if necessary
    project_name="LakeTaupo_example_fish"
    dir_name=os.path.join("sushi", project_name)
    nums=list(range(5,21))
    nums = nums + [29, 38, 131]
    

    im_names=[os.path.join(f"{i}_2024") for i in nums]
    im_paths = [os.path.join(dir_name, id) for id in im_names]  
    exts = ['.jpg', '.jpeg'] # possible image extensions
    
    name_change=args.name_change # defaults to the empty string
    
    if args.single_fish is not None:
        # overwrite to process the specific image
        im_names = [args.single_fish]
        im_paths = [os.path.join(dir_name, args.single_fish)]
    
    if args.cont is not None:
        # continue processing starting with a given fish id
        idx = im_names.index(args.cont)
        im_names=im_names[idx:]
        im_paths=im_paths[idx:]
        
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
            
    if args.preprocess:
        from HandsFreeFishing import preprocess_adult_steelhead_updated
        """
        landmark_length: sets the expected length between the reference points to compute the image scale
        
        ds: stands for down-sample, set to either, 1, 2, 4, or 8. ds=1 is full resolution, ds=8 downsamples by a factor of 8, 
        and will result in lower resolution image displays which can potentiall speed up processing
        
        monitor_idx should only be changed if a multiple monitor display is buing used, and should be adjusted by the user to place pop-up
        windows on the desired monitor
        """
        preprocess_adult_steelhead_updated(im_paths, exts, landmark_length=50,ds=1,monitor_idx=args.monitor)
        
    if args.run:
        from run_scripts import run_adult_steelhead
        run_adult_steelhead(im_paths, dir_name=project_name)
        
    if args.postprocess:
        """ if name_change='', running the postprocessing will overwrite previously computed landmark points with adjusted points,
        and similarly with the landmark point images

        monitor_idx should only be changed if a multiple monitor display is buing used, and should be adjusted by the user to place pop-up
        windows on the desired monitor
        """
        from HandsFreeFishing import postprocess_landmark_points
        landmark_data_dir = os.path.join("landmark_point_data", project_name)
        postprocess_landmark_points(im_paths, project_name, landmark_data_dir, name_change=name_change,monitor_idx=args.monitor)
        
    if args.tps:
        from landmark_points_to_tps import generate_tps_file
        generate_tps_file(im_paths, name_change=name_change, landmark_data_dir='landmark_point_data')
        
        
                
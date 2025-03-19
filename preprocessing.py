import numpy as np
import os
import csv
import regex as re
from tifffile import imread, imwrite
import cv2 as cv

def user_crop_image(image_path=None, image=None, ds=8):
    """Crops an image based on user-selected region."""
    if image_path is None and image is None:
        exit('no image or image path provided')
    
    if image_path is not None:
        # Read the image
        img = cv.imread(image_path)
    else:
        img = image
        
    ds_img = cv.resize(img, (0,0), fx=1/ds, fy=1/ds) 

    # Display the image for user to select the region
    roi = cv.selectROI("Select Region to Crop", ds_img)

    # Crop the image using the selected region
    cropped_img = img[ds*int(roi[1]):ds*int(roi[1]+roi[3]), ds*int(roi[0]):ds*int(roi[0]+roi[2])]

    # Display the cropped image
    # cv.imshow("Cropped Image", cropped_img)
    # cv.waitKey(0)
    cv.destroyAllWindows()

    return cropped_img, ds*np.array(roi)

def splice_im_path(image_path):
    image_path_split = os.path.split(image_path)
    dir = os.path.split(image_path_split[0])[1]
    im_name, ext = os.path.splitext(image_path_split[1])
    
    return dir, im_name, ext

def get_rois_flips_and_bad_paths(im_paths, measurement_dir="measurements", num_fish=None):
        
    rois = []
    horiz_flips = []
    vert_flips = []
    qualities = []
    bad_idxs = []
    
    if num_fish is None:
        num_fish=[1]*len(im_paths)
        
    for (i, im_path) in enumerate(im_paths):
        
        dir, im_name, ext = splice_im_path(im_path)
        
        if num_fish[i]>1:
            idx = re.search(ext, im_path).start()
            new_im_path = im_path[:idx-2]+im_path[idx:] # remove excess labeling
        else:
            new_im_path = im_path
            
        if os.path.exists(new_im_path):
            
            if os.path.exists(os.path.join(measurement_dir, dir, im_name + '.csv')):
                pass # don't overwrite
            else:
                print('\nmade it!\n')
                print(im_path)
                if num_fish[i]>1:
                    idx = re.search(ext, im_path).start()
                    image=cv.imread(new_im_path) # remove excess labeling
                else:
                    image=cv.imread(im_path)
                    
                # copy = np.copy(image)
                
                ds = 3
                cropped_image, roi = user_crop_image(image=image, ds=ds)

                horiz_flip=False
                vertical_flip=False
                
                horiz_flip = input('Enter 0 if the fish is facing left, 1 if right: ')
                if horiz_flip=='1':
                    print('horizontal flip is true')

                vertical_flip = input('\nEnter 0 if the fish is right-side up, 1 if upside down: ')
                if vertical_flip=='1':
                    print('vertical flip is true')
                    
                quality = input('\nEnter 0 for a good quality image, 1 for bad quality: ')
                if quality=='1':
                    print('bad quality is true')

                rois.append(roi)
                horiz_flips.append(horiz_flip)
                vert_flips.append(vertical_flip)
                qualities.append(quality)
                
                if not os.path.exists(os.path.join(measurement_dir,dir)):
                    os.mkdir(os.path.join(measurement_dir,dir))
                    
                with open(os.path.join(measurement_dir, dir, im_name+'.csv'), 'w', newline='') as myfile:
                    wr = csv.writer(myfile)
                    wr.writerow([roi, horiz_flip, vertical_flip, quality])
                    

            
        else:
            bad_idxs.append(i)
            rois.append(None)
            horiz_flips.append(None)
            vert_flips.append(None)
            
    return rois, horiz_flips, vert_flips, bad_idxs

def main():
    from matplotlib import pyplot as plt
    im_path= os.path.join('sushi','example_fish','110524FishID6c.jpg')
    im_paths = [im_path]
    
    if os.path.exists(os.path.join('measurements','example_fish')):
        pass
    else:
        os.mkdir(os.path.join('measurements','example_fish'))
        
    get_rois_flips_and_bad_paths(im_paths)

    
if __name__ == "__main__":
    main()
    
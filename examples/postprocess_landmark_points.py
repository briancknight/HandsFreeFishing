import os
import cv2 as cv
import numpy as np
import pandas as pd

def update_landmark_image(image, landmark_points, dir_name,name='my_fish'):
    os.makedirs(os.path.join("landmark_point_images", dir_name+"_updated"), exist_ok=True)
    cv.imwrite(os.path.join("landmark_point_images", dir_name+"_updated",f'{name}_updated_land_mark_points.png'), image)
    
# dir_name="subset"
# spreadsheet_name="subset"

dir_name="LakeTaupo_example_fish"
spreadsheet_name=''

raw_data_dir = os.path.join("sushi", dir_name)
landmark_data_dir = os.path.join("landmark_point_data", dir_name)
excel_dir = 'spreadsheets'
df = pd.DataFrame()#pd.read_excel(os.path.join(excel_dir, spreadsheet_name+'.xlsx'),sheet_name=0)

if dir_name=="LakeTaupo_example_fish": # define image names directly
    im_names=[os.path.join(f"{i}_2024") for i in [5,6]]
    im_paths = [os.path.join(raw_data_dir, id + ".jpg") for id in im_names]
    landmark_point_paths = [os.path.join(landmark_data_dir,im_name+'_landmark_points.npy') for im_name in im_names]

if dir_name=="subset": # read in data from spreadsheet
    df=pd.read_csv(os.path.join('spreadsheets', spreadsheet_name+'.csv'))
    im_names = [f"{str(df['ID'][i])}_{df['overalldiet'][i]}" for i in range(df.shape[0])]
    im_paths = [os.path.join(raw_data_dir,im_name+'.jpeg') for im_name in im_names]
    landmark_point_paths = [os.path.join(landmark_data_dir,im_name+'_landmark_points.npy') for im_name in im_names]



for idx in range(len(im_paths)):
    # 1. Initialize Landmarks
    landmarks = np.load(landmark_point_paths[idx],allow_pickle=True)
    print(np.shape(landmarks))
    radius = 13
    active_point = None
    img=cv.imread(im_paths[idx])

    def mouse_callback(event, x, y, flags, param):
        global active_point, landmarks
        
        # Left click down: Check if we clicked near a point
        if event == cv.EVENT_LBUTTONDOWN:
            for i, pt in enumerate(landmarks):
                if ((pt[0]-x)**2 + (pt[1]-y)**2) <= radius**2:
                    active_point = i
                    break
                    
        # Mouse move: Move the active point
        elif event == cv.EVENT_MOUSEMOVE:
            if active_point is not None:
                landmarks[active_point] = [x, y]
                
        # Left click up: Release the point
        elif event == cv.EVENT_LBUTTONUP:
            active_point = None
            
        elif event == cv.EVENT_FLAG_RBUTTON:
            pt=[x,y]
            landmarks = np.append(landmarks,[pt],axis=0)
            
    # Bind mouse callback
    cv.namedWindow('Draggable Landmarks')
    cv.setMouseCallback('Draggable Landmarks', mouse_callback,param=[img,active_point,landmarks,radius])
    print(landmarks[:4])
    while True:
        temp_img = img.copy()
        
        # Draw points
        for (i,pt) in enumerate(landmarks.astype(int)):
            cv.circle(temp_img, tuple(pt), radius, (0, 0, 255), -1)
            cv.putText(temp_img, str(i+1), tuple(pt+[0,-10]), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)
            
        cv.imshow('Draggable Landmarks', temp_img)
        if cv.waitKey(1) & 0xFF == 27: # ESC
            break
    cv.destroyAllWindows()

    print(im_names[idx])
    update_landmark_image(temp_img, landmarks, dir_name=dir_name,name=im_names[idx])
    os.makedirs(os.path.join("landmark_point_data",dir_name+'_updated'), exist_ok=True)
    np.save(os.path.join("landmark_point_data",dir_name+'_updated', f"{im_names[idx]}_landmark_points.npy"),landmarks)
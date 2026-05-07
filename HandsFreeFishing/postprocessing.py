import numpy as np
import os
import csv
import regex as re
from tifffile import imread, imwrite
import cv2 as cv
from screeninfo import get_monitors

def get_screen_size():
    for m in get_monitors():
        if m.is_primary:
            screen_width = m.width
            screen_height = m.height
    return screen_width, screen_height

def splice_im_path(image_path):
    image_path_split = os.path.split(image_path)
    dir = os.path.split(image_path_split[0])[1]
    im_name, ext = os.path.splitext(image_path_split[1])
    
    return dir, im_name, ext

def update_landmark_image(image, dir_name,name='my_fish'):
    os.makedirs(os.path.join("landmark_point_images", dir_name+"_updated"), exist_ok=True)
    cv.imwrite(os.path.join("landmark_point_images", dir_name+"_updated",f'{name}_updated_land_mark_points.png'), image)
    
class LandmarkEditor_Post:
    def __init__(self, window_name, image, points, radius=10,ds=1):   
        # meta screen data
        self.ds = ds 
        self.screen_width, self.screen_height = get_screen_size()
        self.window_width = self.ds*int(np.floor(self.screen_width/4))
        self.window_height = self.ds*int(np.floor(self.screen_height/4)) 
         
        # parameters
        self.fontsize = 1.5/self.ds
        self.fontthickness = np.max([int(2/self.ds), 1])
        self.window_name = window_name
        self.image = image
        self.ds_image = cv.resize(self.image, (0,0), fx=1/self.ds, fy=1/self.ds) 
        self.points = points  # List of [x, y]
        self.selected_point_idx = -1
        self.dragging = False
        self.radius = int(radius/self.ds)  # Detection radius for selecting a point
            
    def moving_mouse_event(self, event, x, y, flags, param):
        # 1. Start Dragging: Check if click is near an existing landmark
        if event == cv.EVENT_LBUTTONDOWN:
            for i, (px, py) in enumerate(self.points):
                if np.hypot(px - x, py - y) < self.radius*1.5:
                    self.selected_point_idx = i
                    self.dragging = True
                    break

        # 2. Moving: Update coordinates of the selected landmark
        elif event == cv.EVENT_MOUSEMOVE:
            if self.dragging and self.selected_point_idx != -1:
                self.points[self.selected_point_idx] = [x, y]

        # 3. Stop Dragging: Release the landmark
        elif event == cv.EVENT_LBUTTONUP:
            self.dragging = False
            self.selected_point_idx = -1
         
        # 4. Add additional points by right clicking   
        elif event == cv.EVENT_FLAG_RBUTTON:
            pt=[x,y]
            self.points = np.append(self.points,[pt],axis=0)

    def move_points(self):
        window_name=f"{self.window_name}: ADJUST LANDMARK POINTS"
        cv.namedWindow(window_name)
        cv.setMouseCallback(window_name, self.moving_mouse_event)
        
        while True:
            self.img_display = self.ds_image.copy()
            
            # Draw all landmarks
            for i, (px, py) in enumerate(self.points):
                color = (0, 0, 255) if i == self.selected_point_idx else (0, 255, 0)
                cv.circle(self.img_display, (int(px), int(py)), self.radius, color, -1)
                cv.putText(self.img_display, str(i+1), (int(px) + 10, int(py) - 10), 
                            cv.FONT_HERSHEY_SIMPLEX, self.fontsize, (0, 255, 0), self.fontthickness)

            cv.imshow(window_name, self.img_display)
            
            # Press 'q' to exit or 's' to print current points
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("Current Landmark Points:", self.points)

        cv.destroyAllWindows()
         
    def run(self):
        self.move_points()
  
def postprocess_landmark_points(im_paths,dir_name,landmark_data_dir):
    
    for (idx,im_path) in enumerate(im_paths):
        dir, im_name, ext = splice_im_path(im_path)
        landmark_point_path = os.path.join(landmark_data_dir,im_name+'_landmark_points.npy')
        landmarks = np.load(landmark_point_path,allow_pickle=True)
        radius = 13
        img=cv.imread(im_path)
        
        landmark_post_gui = LandmarkEditor_Post('Draggable Landmarks',img,landmarks,radius=radius)
        landmark_post_gui.run()
        
        landmark_post_gui.points
        
        update_landmark_image(landmark_post_gui.img_display, dir_name=dir_name,name=im_name)
        os.makedirs(os.path.join("landmark_point_data",dir_name+'_updated'), exist_ok=True)
        np.save(os.path.join("landmark_point_data",dir_name+'_updated', f"{im_name}_landmark_points.npy"),landmark_post_gui.points)

        
if __name__=='__main__':
    dir_name="LakeTaupo_example_fish"
    raw_data_dir = os.path.join("..","examples","sushi", dir_name)
    im_names=[os.path.join(f"{i}_2024") for i in [5,6]]
    im_paths = [os.path.join(raw_data_dir, id + ".jpg") for id in im_names]
    landmark_data_dir = os.path.join("..","examples","landmark_point_data", dir_name)
    
    postprocess_landmark_points(im_paths, landmark_data_dir)
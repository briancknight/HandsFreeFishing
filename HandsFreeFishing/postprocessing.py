import numpy as np
import os
import csv
import regex as re
from tifffile import imread, imwrite
import cv2 as cv
from screeninfo import get_monitors
import warnings

def get_screen_size(): # returns width and height of primary monitor
    for m in get_monitors():
        if m.is_primary:
            screen_width = m.width
            screen_height = m.height
    return screen_width, screen_height

def get_monitor(monitor_idx=0):
    for (idx,m) in get_monitors():
        if idx==monitor_idx:
            return m

def splice_im_path(image_path):
    image_path_split = os.path.split(image_path)
    dir = os.path.split(image_path_split[0])[1]
    im_name, ext = os.path.splitext(image_path_split[1])
    
    return dir, im_name, ext

def update_landmark_image(image, dir_name,name='my_fish',name_change=''):
    if name_change=='':
        print(f'\nOverwriting original landmark image for {name}')
    os.makedirs(os.path.join("landmark_point_images", dir_name+name_change), exist_ok=True)
    cv.imwrite(os.path.join("landmark_point_images", dir_name+name_change,f'{name}_landmark_points.png'), image)
    
def update_landmark_points(points, dir_name,name='my_fish',name_change=''):
    if name_change=='':
        print(f'Overwriting original landmark points for {name}\n')
    os.makedirs(os.path.join("landmark_point_data", dir_name+name_change), exist_ok=True)
    np.save(os.path.join("landmark_point_data", dir_name+name_change,f'{name}_landmark_points.npy'), points)
    
# def update_landmark_image(points, dir_name,name='my_fish'):
#     os.makedirs(os.path.join("landmark_point_data",dir_name+'_updated'), exist_ok=True)
#     np.save(os.path.join("landmark_point_data",dir_name+'_updated', f"{im_name}_landmark_points.npy"),landmark_post_gui.points)
    
class LandmarkEditor_Post:
    def __init__(self, window_name, image, points, radius=10,ds=1,monitor_idx=0):   
        # meta screen data
        self.ds = ds 
        monitors = get_monitors()
        if monitor_idx > len(monitors) - 1:
            warnings.warn('Monitor index is too large, defaulting to 0')
            monitor_idx=0
            
        self.monitor_idx=monitor_idx
        self.monitor=monitors[monitor_idx]
        print(f"current monitor={self.monitor}")
        self.screen_width, self.screen_height = (self.monitor.width,self.monitor.height)
        self.window_width = self.ds*int(self.screen_width)
        self.window_height = self.ds*int(self.screen_height) 
         
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
            try:
                self.points = np.append(self.points,[pt],axis=0)
            except ValueError:
                self.points=[pt]

    def move_points(self):
        window_name=f"{self.window_name}: ADJUST LANDMARK POINTS"
        
        if self.monitor.is_primary:
            cv.namedWindow(window_name, cv.WINDOW_FULLSCREEN)
        else:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            cv.moveWindow(window_name, self.monitor.x, 0)
            # cv.setWindowProperty(window_name,cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        # cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        # cv.moveWindow(window_name, self.monitor.width, 0)
        # cv.setWindowProperty(window_name,cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)    
        
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
  
def postprocess_landmark_points(im_paths,dir_name,landmark_data_dir,name_change='_updated',monitor_idx=0, exts=['.jpg','.jpeg','.png']):
    
    for (idx,im_path) in enumerate(im_paths):
        
        found_im_path=False
        for ext in exts:
            im_path_temp=im_path+ext
            if os.path.exists(im_path_temp):
                im_path = im_path_temp
                found_im_path=True
                break
        
        if not found_im_path:
            print(f"Failed to find image path: {im_path}")

        else:
            
            dir, im_name, ext = splice_im_path(im_path)
            landmark_point_path = os.path.join(landmark_data_dir,im_name+'_landmark_points.npy')
            if os.path.exists(landmark_point_path):
                landmarks = np.load(landmark_point_path,allow_pickle=True)
            else:
                print(f"No landmark points found for: {im_path}, please manually place points instead")
                landmarks = []
            
            radius = 13
            img=cv.imread(im_path)
            
            landmark_post_gui = LandmarkEditor_Post('Draggable Landmarks',img,landmarks,radius=radius,monitor_idx=monitor_idx)
            landmark_post_gui.run()
            
            landmark_post_gui.points
            
            update_landmark_image(landmark_post_gui.img_display, dir_name=dir_name,name=im_name,name_change=name_change)
            update_landmark_points(landmark_post_gui.points, dir_name=dir_name, name=im_name,name_change=name_change)
            # update_landmark_points(landmark_post_gui.points, dir_name=dir_name, name=im_name)
            os.makedirs(os.path.join("landmark_point_data",dir_name+'_updated'), exist_ok=True)
            np.save(os.path.join("landmark_point_data",dir_name+'_updated', f"{im_name}_landmark_points.npy"),landmark_post_gui.points)

        
if __name__=='__main__':
    dir_name="LakeTaupo_example_fish"
    raw_data_dir = os.path.join("..","examples","sushi", dir_name)
    im_names=[os.path.join(f"{i}_2024") for i in [5,6]]
    im_paths = [os.path.join(raw_data_dir, id + ".jpg") for id in im_names]
    landmark_data_dir = os.path.join("..","examples","landmark_point_data", dir_name)
    
    postprocess_landmark_points(im_paths, landmark_data_dir)
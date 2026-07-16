import numpy as np
import os
import csv
import regex as re
from tifffile import imread, imwrite
import cv2 as cv
from screeninfo import get_monitors
import warnings

def get_screen_size():
    for m in get_monitors():
        if m.is_primary:
            screen_width = m.width
            screen_height = m.height
    return screen_width, screen_height
        
class LandmarkEditor:
    def __init__(self, window_name, image, points, point_names=None, box_names = None, radius=10, landmark_length=50,ds=1,monitor_idx=0):   
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
        self.zoom_scale=1.0
        self.center_x, self.center_y = image.shape[1]//2, image.shape[0]//2
        self.fontsize = 1.5/self.ds
        self.fontthickness = np.max([int(2/self.ds), 1])
        self.window_name = window_name
        self.image = image
        self.ds_image = cv.resize(self.image, (0,0), fx=1/self.ds, fy=1/self.ds) 
        self.points = points  # List of [x, y]
        self.selected_point_idx = -1
        self.dragging = False
        self.radius = int(radius/self.ds)  # Detection radius for selecting a point
        self.landmark_length = landmark_length
        self.MAX_POINTS = 8
        self.MAX_BOXES = 3
        self.point_names = point_names
        if self.point_names is None:
            self.point_names = [str(i) for i in range(1,self.MAX_POINTS+1)]
            
        self.box_names = box_names
        if self.box_names is None:
            self.box_names = [str(i) for i in range(1,self.MAX_BOXES+1)]

    def mouse_callback_n_points(self, event, x, y, flags, param):
        """
        Mouse callback function to capture landmark points.
        """
        
        if event == cv.EVENT_LBUTTONDOWN:
            if len(self.points) < self.MAX_POINTS:
                # Store the point coordinates
                self.points.append((x, y))
                if len(self.points) == self.MAX_POINTS:
                    print(f"{self.MAX_POINTS} landmark points have been captured.")
                    # Optional: Perform an action here after getting both points
                    # e.g., calculate distance, crop ROI, etc.
        # elif event == cv.EVENT_MOUSEWHEEL:
        #     # Increase or decrease scale factor based on scroll direction
        #     if flags > 0: 
        #         self.zoom_scale += 0.01
        #     else: 
        #         self.zoom_scale -= 0.01
        #     self.center_x, self.center_y = x, y
            
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

    def moving_crop_event(self, event, x, y, flags, param):
        # 1. Start Dragging: Check if click is near an existing landmark
        if event == cv.EVENT_LBUTTONDOWN:
            for i, (px, py, w, h) in enumerate(self.rois):
                if np.hypot(px - x, py - y) < self.radius*1.5:
                    self.selected_roi_idx = i
                    break

        # # 2. Moving: Update coordinates of the selected landmark
        # elif event == cv.EVENT_MOUSEMOVE:
        #     if self.dragging and self.selected_roi_idx != -1:
        #         self.rois[self.selected_point_idx] = [x, y]

        # # 3. Stop Dragging: Release the landmark
        # elif event == cv.EVENT_LBUTTONUP:
        #     self.dragging = False
        #     self.selected_point_idx = -1
            
    def select_points(self):
        # Setup OpenCV window and mouse callback
        window_name=f"{self.window_name}: PLACE LANDMARK POINTS"
        if self.monitor.is_primary:
            cv.namedWindow(window_name, cv.WINDOW_FULLSCREEN)
        else:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            cv.moveWindow(window_name, self.monitor.x-1, self.monitor.y-1)
            cv.setWindowProperty(window_name,cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)   
            
        # cv.resizeWindow(window_name, self.window_width, self.window_height)
        cv.setMouseCallback(window_name, self.mouse_callback_n_points)
        
        while len(self.points)<self.MAX_POINTS:
            img_display = self.ds_image.copy()
            # h, w = img_display.shape[:2]
            # # Calculate new dimensions and display
            # # self.zoom_scale = max(0.1, self.zoom_scale)
            # # self.zoom_scale = min(self.zoom_scale, 2)
            # new_h, new_w = int(h / self.zoom_scale), int(w / self.zoom_scale)
            # # Calculate top-left corner of the crop (clamped to image boundaries)
            # start_y = int(self.center_y/self.zoom_scale) - int(new_h/2)
            # start_x = int(self.center_x/self.zoom_scale) - int(new_w/2)
            # Draw all landmarks
            for i, (px, py) in enumerate(self.points):
                color = (0, 0, 255) if i == self.selected_point_idx else (255, 0, 0)
                cv.circle(img_display, (px, py), self.radius, color, -1)
                cv.putText(img_display, self.point_names[i], (px + 10, py - 10), 
                            cv.FONT_HERSHEY_SIMPLEX, self.fontsize, (0, 255, 0), self.fontthickness)

            # Slice and resize
            # crop = img_display[start_y:start_y + new_h, start_x:start_x + new_w]
            # res = cv.resize(crop, (w, h), interpolation=cv.INTER_LINEAR)
            # cv.imshow(window_name, res)
            cv.imshow(window_name, img_display)
            
            # Press 'q' to exit or 's' to print current points
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("Current Landmark Points:", self.points)

        cv.destroyAllWindows()
        
    def move_points(self):
        window_name=f"{self.window_name}: ADJUST LANDMARK POINTS"
        
        if self.monitor.is_primary:
            cv.namedWindow(window_name, cv.WINDOW_FULLSCREEN)
        else:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            cv.moveWindow(window_name, self.monitor.x-1, self.monitor.y-1)
            cv.setWindowProperty(window_name,cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)    
            
        cv.setMouseCallback(window_name, self.moving_mouse_event)
        
        while True:
            img_display = self.ds_image.copy()
            
            # Draw all landmarks
            for i, (px, py) in enumerate(self.points):
                color = (0, 0, 255) if i == self.selected_point_idx else (0, 255, 0)
                cv.circle(img_display, (px, py), self.radius, color, -1)
                cv.putText(img_display, self.point_names[i], (px + 10, py - 10), 
                            cv.FONT_HERSHEY_SIMPLEX, self.fontsize, (0, 255, 0), self.fontthickness)

            cv.imshow(window_name, img_display)
            
            # Press 'q' to exit or 's' to print current points
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("Current Landmark Points:", self.points)

        cv.destroyAllWindows()
         
    def user_crop_image(self):
        """Gathers bounding box from an image based on user-selected region."""

        # Display the image for user to select the region
        window_name = f"{self.window_name}: DRAW A BOUNDING BOX AROUND THE {self.box_names[0].upper()},{self.box_names[1].upper()}, {self.box_names[2].upper()}"
        
        if self.monitor.is_primary:
            cv.namedWindow(window_name, cv.WINDOW_FULLSCREEN)
        else:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            cv.moveWindow(window_name, self.monitor.x-1, self.monitor.y-1)
            cv.setWindowProperty(window_name,cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)   
            
        self.rois = []
        img_display = self.ds_image.copy()
        
        while True:
            
            if len(self.rois) < 3:  
                self.rois.append(np.array(cv.selectROI(window_name, img_display)))     
            # Draw all landmarks
            for i, rect in enumerate(self.rois):
                x, y, w, h = rect
                # Use NumPy slicing to crop the image: image[y:y+h, x:x+w]
                cv.rectangle(img_display, (x,y), (x+w,y+h), (255,0,0),thickness=3)
                cv.putText(img_display, self.box_names[i], (x + 10, y - 10), 
                            cv.FONT_HERSHEY_SIMPLEX, self.fontsize, (0, 255, 0), self.fontthickness)
            
            
            
            if len(self.rois)==3:
                break
                # window_name=f"{self.window_name}: DRAW A BOUNDING BOX AROUND THE {self.box_names[len(self.rois)].upper()}"
            cv.imshow(window_name, img_display)
            # key = cv.waitKey(1) & 0xFF
            
    
        cv.destroyAllWindows() 

        return self.rois
    
    def redo_crop(self):
        window_name=f"{self.window_name}: ADJUST BOUNDING BOXES"
        if self.monitor.is_primary:
            cv.namedWindow(window_name, cv.WINDOW_FULLSCREEN)
        else:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            cv.moveWindow(window_name, self.monitor.x-1, self.monitor.y-1)
            cv.setWindowProperty(window_name,cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)   
        
        cv.setMouseCallback(window_name, self.moving_crop_event)
        self.selected_roi_idx=-1
 
        while True:
            img_display = self.ds_image.copy()
            # Draw all landmarks
            for i, (px, py, w, h) in enumerate(self.rois):
                color = (0, 0, 255) if i == self.selected_roi_idx else (0, 255, 0)
                cv.rectangle(img_display, (px, py), (px+w,py+h), color,thickness=3)
                cv.putText(img_display, self.box_names[i], (px + 10, py - 10), 
                            cv.FONT_HERSHEY_SIMPLEX, self.fontsize, color, self.fontthickness)

            cv.imshow(window_name, img_display)
                
            # Press 'q' to exit or 's' to print current points
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if self.selected_roi_idx != -1:
                    new_roi = cv.selectROI(window_name, img_display)
                    self.rois[self.selected_roi_idx] = np.array(new_roi)
                    cv.setMouseCallback(window_name, self.moving_crop_event)
                
        cv.destroyAllWindows()
        
    def run(self):
        self.select_points()
        self.move_points()
        self.user_crop_image()
        self.redo_crop()
        
        self.rois = [self.ds*np.array(roi) for roi in self.rois]
        self.points = [self.ds*np.array(point) for point in self.points]
        
    def get_scale(self):
        point1 = self.points[0]
        point2 = self.points[1]
        
        self.pixels_per_mm = np.round(np.linalg.norm(np.array(point1) - np.array(point2))/self.landmark_length,2)

class LandmarkEditor_juvenile(LandmarkEditor):
    def __init__(self, window_name, image, 
                 points,point_names=None, box_names = None, 
                 radius=10, landmark_length=50,ds=1,monitor_idx=0):
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
        self.zoom_scale=1.0
        self.center_x, self.center_y = image.shape[1]//2, image.shape[0]//2
        self.fontsize = 1.5/self.ds
        self.fontthickness = np.max([int(2/self.ds), 1])
        self.window_name = window_name
        self.image = image
        self.ds_image = cv.resize(self.image, (0,0), fx=1/self.ds, fy=1/self.ds) 
        self.points = points  # List of [x, y]
        self.selected_point_idx = -1
        self.dragging = False
        self.radius = int(radius/self.ds)  # Detection radius for selecting a point
        self.landmark_length = landmark_length
        self.MAX_POINTS = len(point_names)
        self.MAX_BOXES = len(box_names)
        self.point_names = point_names
        if self.point_names is None:
            self.point_names = [str(i) for i in range(1,self.MAX_POINTS+1)]
            
        self.box_names = box_names
        if self.box_names is None:
            self.box_names = [str(i) for i in range(1,self.MAX_BOXES+1)]
    
    def user_crop_image(self):
        """Gathers bounding box from an image based on user-selected region."""

        # Display the image for user to select the region
        window_name = f"{self.window_name}: DRAW A BOUNDING BOX AROUND THE {self.box_names[0].upper()}"
        
        if self.monitor.is_primary:
            cv.namedWindow(window_name, cv.WINDOW_FULLSCREEN)
        else:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            cv.moveWindow(window_name, self.monitor.x-1, self.monitor.y-1)
            cv.setWindowProperty(window_name,cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)   
            
        self.rois = []
        img_display = self.ds_image.copy()
        
        while True:
            
            if len(self.rois) < self.MAX_BOXES:  
                self.rois.append(np.array(cv.selectROI(window_name, img_display)))     
            # Draw all landmarks
            for i, rect in enumerate(self.rois):
                x, y, w, h = rect
                # Use NumPy slicing to crop the image: image[y:y+h, x:x+w]
                cv.rectangle(img_display, (x,y), (x+w,y+h), (255,0,0),thickness=3)
                cv.putText(img_display, self.box_names[i], (x + 10, y - 10), 
                            cv.FONT_HERSHEY_SIMPLEX, self.fontsize, (0, 255, 0), self.fontthickness)
            
            
            
            if len(self.rois)==self.MAX_BOXES:
                break
                # window_name=f"{self.window_name}: DRAW A BOUNDING BOX AROUND THE {self.box_names[len(self.rois)].upper()}"
            cv.imshow(window_name, img_display)
            # key = cv.waitKey(1) & 0xFF
            
    
        cv.destroyAllWindows() 

        return self.rois
    
    def run(self):
        self.select_points()
        self.move_points()
        self.user_crop_image()
        self.redo_crop()
        
        self.rois = [self.ds*np.array(roi) for roi in self.rois]
        self.points = [self.ds*np.array(point) for point in self.points]
        
def mouse_callback_n_points(event, x, y, flags, param):
    """
    Mouse callback function to capture landmark points.
    """
    img=param[0]
    landmark_points=param[1]
    MAX_POINTS = param[2]
    fin_markers=param[3]
    fin_names=['dorsal', 'adipose', 'caudal', 'anal', 'pelvic', 'pectoral']
    
    if event == cv.EVENT_LBUTTONDOWN:
        if len(landmark_points) < MAX_POINTS:
            # Store the point coordinates
            landmark_points.append((x, y))
            # Draw a circle on the image to mark the point
            cv.circle(img, (x, y), 10, (0, 255, 0), -1)
            if fin_markers:
                cv.putText(img, fin_names[len(landmark_points)-1], (x+10,y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)
            # cv.imshow("Image with Landmarks", img)
            print(f"Point {len(landmark_points)} captured at: ({x}, {y})")
            print(f"length of landmark points=",{len(landmark_points)})
            if len(landmark_points) == MAX_POINTS:
                print(f"{MAX_POINTS} landmark points have been captured.")
                # Optional: Perform an action here after getting both points
                # e.g., calculate distance, crop ROI, etc.

def mouse_callback_n_points_move(event, x, y, flags, param):
    """
    Mouse callback function to capture landmark points.
    """
    fin_point_dict=param[0]
    radius=param[1]
    active_point=None
    fin_names=['dorsal', 'adipose', 'caudal', 'anal', 'pelvic', 'pectoral']
    landmarks=[fin_point_dict[name] for name in fin_names]
    if event == cv.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(landmarks):
            if ((pt[0]-x)**2 + (pt[1]-y)**2) <= radius**2:
                active_point = i
                break
                        
    # Mouse move: Move the active point
    elif event == cv.EVENT_MOUSEMOVE:
        if active_point is not None:
            fin_point_dict[fin_names[i]] = [x, y]
        
    # Left click up: Release the point
    elif event == cv.EVENT_LBUTTONUP:
        active_point = None
    
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
        
        # create new point with right click 
        elif event == cv.EVENT_FLAG_RBUTTON:
            pt=[x,y]
            landmarks = np.append(landmarks,[pt],axis=0)
                               
def user_reference_scale(img_path=None,img=None,landmark_length=5):
    if img_path is None and img is None:
        IndexError("Please include either an image path or image")
    
    if img is None: # Use loaded image unless there is none
        img=cv.imread(img_path)
        
    landmark_points=[]
    MAX_POINTS=2
    window_name=f"Place scale reference points {landmark_length} mm apart"
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, mouse_callback_n_points,param=[img,landmark_points, MAX_POINTS, False])
    
    print("Click on two points in the image window.")

    # Display the image and wait for a key press
    while True:
        cv.imshow(window_name, img)
        if cv.waitKey(1) & 0xFF == 27: # Press 'ESC' to exit
            break
        if len(landmark_points) == MAX_POINTS:
            # Stop waiting for input after two points are collected
            # unless you want to allow changing the points
            break

    cv.destroyAllWindows()
    
    # The captured points are stored in the landmark_points list
    if len(landmark_points) == MAX_POINTS:
        point1 = landmark_points[0]
        point2 = landmark_points[1]
        pixels_per_mm = np.round(np.linalg.norm(np.array(point1) - np.array(point2))/landmark_length,2)
        # print(f"Final Point 1: {point1}, Point 2: {point2}")
        # print("approximately ", pixels_per_mm, "pixels/mm")
    return pixels_per_mm

def user_fin_points(img_path=None,img=None,n_points=6):
    if img_path is None and img is None:
        IndexError("Please include either an image path or image")
    
    if img is None: # Use loaded image unless there is none
        img=cv.imread(img_path)
        
    landmark_points=[]
    MAX_POINTS=n_points
    cv.namedWindow("Fin Markers")
    cv.setMouseCallback("Fin Markers", mouse_callback_n_points,param=[img,landmark_points, MAX_POINTS,True])
    
    print("Click on two points in the image window.")

    # Display the image and wait for a key press
    while True:
        cv.imshow("Fin Markers", img)
        if cv.waitKey(1) & 0xFF == 27: # Press 'ESC' to exit
            break
        if len(landmark_points) == MAX_POINTS:
            ## PREVIOUS VERSION
            ## Stop waiting for input after MAX_POINTS points are collected
            ## unless you want to allow changing the points
            # break
            
            ## UPDATED VERSION
            # after MAX_POINTS are collected, allow user to move onto next phase by pressing escape key
            if cv.waitKey(1) & 0xFF == 27: # ESC
                break

    cv.destroyAllWindows()
    if len(landmark_points) == MAX_POINTS:
        return landmark_points

def user_crop_image(image_path=None, image=None, ds=8, to_crop=None):
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
    if to_crop is not None:
        window_name = f"Please draw a bounding box around the {to_crop}."
    else: 
        window_name = "Select Region to Crop"
    screen_width, screen_height = get_screen_size()
    window_width = int(np.floor(screen_width/4))
    window_height = int(np.floor(screen_height/4))
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    
    cv.resizeWindow(window_name, window_width, window_height)
    roi = cv.selectROI(window_name, ds_img)

    # Crop the image using the selected region
    cropped_img = img[ds*int(roi[1]):ds*int(roi[1]+roi[3]), ds*int(roi[0]):ds*int(roi[0]+roi[2])]

    # Display the cropped image
    # cv.imshow("Cropped Image", cropped_img)
    # cv.waitKey(0)
    cv.destroyAllWindows()

    return cropped_img, ds*np.array(roi)

# landmark_points = []
# Maximum number of points to select
# MAX_POINTS = 2
                
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
                
                horiz_flip = input('Enter 0 if the fish is facing left, 1 if right: ')
                if horiz_flip=='1':
                    print('horizontal flip is true')
                else:
                    horiz_flip='0'

                vertical_flip = input('\nEnter 0 if the fish is right-side up, 1 if upside down: ')
                if vertical_flip=='1':
                    print('vertical flip is true')
                else:
                    vertical_flip='0'
                    
                quality = input('\nEnter 0 for a good quality image, 1 for bad quality: ')
                if quality=='1':
                    print('bad quality is true')
                else:
                    quality='0'

                rois.append(roi)
                horiz_flips.append(horiz_flip)
                vert_flips.append(vertical_flip)
                qualities.append(quality)
                
                if not os.path.exists(os.path.join(measurement_dir,dir)):
                    os.makedirs(os.path.join(measurement_dir,dir),exist_ok=True)
                    
                with open(os.path.join(measurement_dir, dir, im_name+'.csv'), 'w', newline='') as myfile:
                    wr = csv.writer(myfile)
                    wr.writerow([roi, horiz_flip, vertical_flip, quality])
                    

            
        else:
            bad_idxs.append(i)
            rois.append(None)
            horiz_flips.append(None)
            vert_flips.append(None)
            
    return rois, horiz_flips, vert_flips, bad_idxs

def preprocess_adult_steelhead(im_paths, measurement_dir="measurements", num_fish=None, landmark_length=50):
        
    scales = []
    eye_rois = []
    head_rois = []
    rois = []
    horiz_flips = []
    vert_flips = []
    qualities = []
    bad_idxs = []
    
    if num_fish is None:
        num_fish=[1]*len(im_paths)
        
    for (i, im_path) in enumerate(im_paths):
        
        print(im_path)
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
                
                ds = 1
                img_copy=image.copy()
                pixels_per_mm = user_reference_scale(img=img_copy, landmark_length=landmark_length)
                fin_points = user_fin_points(img=img_copy, n_points=6)
                fin_names=['dorsal', 'adipose', 'caudal', 'anal', 'pelvic', 'pectoral']
                # fin_points = user_fin_points(img=image, num_points=6)

                fin_point_editor = LandmarkEditor("Fin Point Editor", image, fin_points, point_names=fin_names)
                fin_point_editor.run()         
                       
                _, eye_roi = user_crop_image(image=image, ds=ds, to_crop="eyeball")
                _, head_roi = user_crop_image(image=image, ds=ds, to_crop="head")
                cropped_image, roi = user_crop_image(image=image, ds=ds)
                
                horiz_flip = input('Enter 0 if the fish is facing left, 1 if right: ')
                if horiz_flip=='1':
                    print('horizontal flip is true')
                else:
                    horiz_flip='0'

                vertical_flip = input('\nEnter 0 if the fish is right-side up, 1 if upside down: ')
                if vertical_flip=='1':
                    print('vertical flip is true')
                else:
                    vertical_flip='0'
                    
                quality = input('\nEnter 0 for a good quality image, 1 for bad quality: ')
                if quality=='1':
                    print('bad quality is true')
                else:
                    quality='0'
                scales.append(1/pixels_per_mm)
                eye_rois.append(eye_roi)
                head_rois.append(head_roi)
                rois.append(roi)
                horiz_flips.append(horiz_flip)
                vert_flips.append(vertical_flip)
                qualities.append(quality)
                
                if not os.path.exists(os.path.join(measurement_dir,dir)):
                    os.makedirs(os.path.join(measurement_dir,dir),exist_ok=True)
                
                np.save(os.path.join(measurement_dir, dir, im_name+'_fin_points.npy'), np.array(fin_points))
                    
                with open(os.path.join(measurement_dir, dir, im_name+'.csv'), 'w', newline='') as myfile:
                    wr = csv.writer(myfile)
                    wr.writerow([1/pixels_per_mm, eye_roi, head_roi, roi, horiz_flip, vertical_flip, quality])
                    

            
        else:
            bad_idxs.append(i)
            scales.append(None)
            rois.append(None)
            horiz_flips.append(None)
            vert_flips.append(None)
            qualities.append(None)
            
    return rois, horiz_flips, vert_flips, bad_idxs

def preprocess_adult_steelhead_updated(im_paths, exts, measurement_dir="measurements", num_fish=None, landmark_length=50, orientation_prompts=False,ds=1,monitor_idx=0):
        
    scales = []
    eye_rois = []
    head_rois = []
    rois = []
    horiz_flips = []
    vert_flips = []
    qualities = []
    bad_idxs = []
    
        
    for (i, im_path) in enumerate(im_paths):
        
        print(im_path)
        
        found_im_path=False
        for ext in exts:
            im_path_temp=im_path+ext
            if os.path.exists(im_path_temp):
                im_path = im_path_temp
                found_im_path=True
                break
            
        if not found_im_path:
            print(f"Failed to find image path: {im_path}")
            bad_idxs.append(i)
            scales.append(None)
            rois.append(None)
            horiz_flips.append(None)
            vert_flips.append(None)
            qualities.append(None)
        else:
            dir, im_name, ext = splice_im_path(im_path)            
            if os.path.exists(os.path.join(measurement_dir, dir, im_name + '.csv')):
                pass # don't overwrite
            else:
                print('\nmade it!\n')
                print(im_path)

                image=cv.imread(im_path)
                                    
                img_copy=image.copy()
                point_names=['scale pt 1', 'scale pt 2', 'dorsal', 'adipose', 'caudal', 'anal','pelvic', 'pectoral']
                box_names = ['eyeball', 'head', 'fish']
                radius=13
                print(img_copy.shape)
                landmark_gui = LandmarkEditor(im_path, img_copy, [], 
                                              point_names=point_names,box_names=box_names,
                                              radius=radius, landmark_length=landmark_length,ds=ds,
                                              monitor_idx=monitor_idx)
                landmark_gui.run()
                landmark_gui.get_scale()
                
                pixels_per_mm = landmark_gui.pixels_per_mm
                fin_points = landmark_gui.points[2:]
                       
                eye_roi = landmark_gui.rois[0]
                head_roi = landmark_gui.rois[1]
                roi = landmark_gui.rois[2]
                
                if orientation_prompts:
                    horiz_flip = input('Enter 0 if the fish is facing left, 1 if right: ')
                    if horiz_flip=='1':
                        print('horizontal flip is true')
                    else:
                        horiz_flip='0'

                    vertical_flip = input('\nEnter 0 if the fish is right-side up, 1 if upside down: ')
                    if vertical_flip=='1':
                        print('vertical flip is true')
                    else:
                        vertical_flip='0'
                        
                    quality = input('\nEnter 0 for a good quality image, 1 for bad quality: ')
                    if quality=='1':
                        print('bad quality is true')
                    else:
                        quality='0'
                else:
                    horiz_flip='0'
                    vertical_flip='0'
                    quality='0'
                    
                scales.append(1/pixels_per_mm)
                eye_rois.append(eye_roi)
                head_rois.append(head_roi)
                rois.append(roi)
                horiz_flips.append(horiz_flip)
                vert_flips.append(vertical_flip)
                qualities.append(quality)
                
                if not os.path.exists(os.path.join(measurement_dir,dir)):
                    os.makedirs(os.path.join(measurement_dir,dir),exist_ok=True)
                
                np.save(os.path.join(measurement_dir, dir, im_name+'_fin_points.npy'), np.array(fin_points))
                    
                with open(os.path.join(measurement_dir, dir, im_name+'.csv'), 'w', newline='') as myfile:
                    wr = csv.writer(myfile)
                    wr.writerow([1/pixels_per_mm, eye_roi, head_roi, roi, horiz_flip, vertical_flip, quality])
        
            
    return rois, horiz_flips, vert_flips, bad_idxs

def preprocess_juvenile_steelhead(im_paths, exts, genetic_ids, life_stages, measurement_dir="measurements", project_dir = None, num_fish=None, landmark_length=50, orientation_prompts=False,ds=1,monitor_idx=0):
        
    scales = []
    # eye_rois = []
    # head_rois = []
    yolk_sac_rois = []
    rois = []
    horiz_flips = []
    vert_flips = []
    qualities = []
    bad_idxs = []
    
    
    for (i, im_path) in enumerate(im_paths):
                    
        print(im_path)
        
        found_im_path=False
        for ext in exts:
            im_path_temp=im_path+ext
            if os.path.exists(im_path_temp):
                im_path = im_path_temp
                found_im_path=True
                break
            
        if not found_im_path:
            print(f"Failed to find image path: {im_path}")
            bad_idxs.append(i)
            scales.append(None)
            rois.append(None)
            horiz_flips.append(None)
            vert_flips.append(None)
            qualities.append(None)
        else:

            dir, im_name, ext = splice_im_path(im_path)     
                       
            if os.path.exists(os.path.join(measurement_dir, project_dir, im_name + '.csv')):
                pass # don't overwrite
            else:
                print('\nmade it!\n')
                print(f"preprocessing im_path with genetic id {genetic_ids[i]}")
                print(f"life stage is: {life_stages[i].lower()}")

                image=cv.imread(im_path)
                                    
                img_copy=image.copy()
                is_hatched = (life_stages[i].strip().lower() == 'hatched')
                print(is_hatched)
                
                if is_hatched:
                    point_names=['scale pt 1', 'scale pt 2','eyeball', 'yolk sac']
                    box_names = ['fish', 'yolk sac']
                else:
                    point_names=['scale pt 1', 'scale pt 2','eyeball']
                    box_names = ['fish']
                    
                print(point_names)
                radius=13
                landmark_gui = LandmarkEditor_juvenile(im_path, img_copy, [], 
                                              point_names=point_names,box_names=box_names,
                                              radius=radius, landmark_length=landmark_length,ds=ds,
                                              monitor_idx=monitor_idx)
                landmark_gui.run()
                landmark_gui.get_scale()
                
                pixels_per_mm = landmark_gui.pixels_per_mm
                fin_points = landmark_gui.points[2:]
                
                if is_hatched:
                    roi = landmark_gui.rois[0]
                    yolk_sac_roi = landmark_gui.rois[1]    
                else:
                    yolk_sac_roi = None
                    roi = landmark_gui.rois[0]
                
                
                if orientation_prompts:
                    horiz_flip = input('Enter 0 if the fish is facing left, 1 if right: ')
                    if horiz_flip=='1':
                        print('horizontal flip is true')
                    else:
                        horiz_flip='0'

                    vertical_flip = input('\nEnter 0 if the fish is right-side up, 1 if upside down: ')
                    if vertical_flip=='1':
                        print('vertical flip is true')
                    else:
                        vertical_flip='0'
                        
                    quality = input('\nEnter 0 for a good quality image, 1 for bad quality: ')
                    if quality=='1':
                        print('bad quality is true')
                    else:
                        quality='0'
                else:
                    horiz_flip='0'
                    vertical_flip='0'
                    quality='0'
                    
                scales.append(1/pixels_per_mm)
                # eye_rois.append(eye_roi)
                # head_rois.append(head_roi)
                yolk_sac_rois.append(yolk_sac_roi)
                rois.append(roi)
                horiz_flips.append(horiz_flip)
                vert_flips.append(vertical_flip)
                qualities.append(quality)
                
                if not os.path.exists(os.path.join(measurement_dir,project_dir)):
                    os.makedirs(os.path.join(measurement_dir,project_dir),exist_ok=True)
                
                np.save(os.path.join(measurement_dir, project_dir, im_name+'_fin_points.npy'), np.array(fin_points))
                    
                with open(os.path.join(measurement_dir, project_dir, im_name+'.csv'), 'w', newline='') as myfile:
                    wr = csv.writer(myfile)
                    wr.writerow([1/pixels_per_mm, roi, yolk_sac_roi, horiz_flip, vertical_flip, quality])
        
            
    return rois, horiz_flips, vert_flips, bad_idxs

def main():
    
    from matplotlib import pyplot as plt
    im_path= os.path.join('sushi','example_fish','110524FishID6c.jpg')
    im_paths = [im_path]
    
    if os.path.exists(os.path.join('measurements','example_fish')):
        pass
    else:
        os.makedirs(os.path.join('measurements','example_fish'),exist_ok=True)
        
    get_rois_flips_and_bad_paths(im_paths)

def testing():
    from matplotlib import pyplot as plt
    img_path=os.path.join("..","examples", "sushi", "LakeTaupo_example_fish", "5_2024.JPG")
    # user_reference_scale(img_path,landmark_length=25)
    # _,_ = user_crop_image(img_path,ds=1)
    preprocess_adult_steelhead([img_path],measurement_dir=os.path.join("..","examples","measurements"))


if __name__ == "__main__":
    testing() #main()


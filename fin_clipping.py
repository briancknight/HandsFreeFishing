import numpy as np
import os
import csv
from tifffile import imread, imwrite
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import cv2 as cv
from pyefd import elliptic_fourier_descriptors, calculate_dc_coefficients, plot_efd, reconstruct_contour
from segment_anything import SamPredictor, sam_model_registry
import pandas as pd
import regex as re

# helpers

def get_largest_connected_component(mask):
    """
    Extracts the largest connected component from a binary mask.

    Args:
        mask (numpy.ndarray): A binary mask (0s and 1s or True/False).

    Returns:
        numpy.ndarray: A mask containing only the largest connected component,
                       or an empty mask if no components are found.
    """
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(mask)

    largest_component_id = 1
    largest_area = stats[1, cv.CC_STAT_AREA]

    for i in range(2, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_component_id = i

    largest_component_mask = np.zeros_like(mask, dtype=np.uint8)
    largest_component_mask[labels == largest_component_id] = 255

    return largest_component_mask

def get_grid_scale(img):
    """Computes the relative scale of the image by finding the best match for a template image of a square grid of various sizes

    Args:
        img (_type_): _description_

    Returns:
        scale (int): the number of pixels per grid square in the given image
        top_left (np.ndarray): coordinates of top left corner of best match
        bottom_right (np.ndarray): coordinates of bottom right corner of best match
    """
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    template = cv.imread('templates/IMG_2937_grid_template.tif', cv.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]
    
    meth = 'TM_CCOEFF_NORMED'
    sizes = np.linspace(0.25, 3, 50)
    # sizes = [1, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4, 0.6, 1.5, 0.5]
    
    min_vals = []
    max_vals = []
    min_locs = []
    max_locs = []
    ress = []
    tls=[]
    brs=[]
    
    for sz in sizes:

        tmp_template = cv.resize(template, (0,0), fx = sz, fy = sz)
        w, h = tmp_template.shape[::-1]
        method = getattr(cv, meth)
    
        # Apply template Matching
        res = cv.matchTemplate(img, tmp_template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        
    
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:    
            top_left = min_loc
        else:             
            top_left = max_loc
        
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        min_locs.append(min_loc)
        max_locs.append(max_loc)
        min_vals.append(min_val)
        max_vals.append(max_val)
        ress.append(res)
        tls.append(top_left)
        brs.append(bottom_right)
        
    max_vals=np.array(max_vals)
    min_vals=np.array(min_vals)
    best_idx = np.argmax(max_vals)
    
    res = ress[best_idx]
    bottom_right = brs[best_idx]
    top_left = tls[best_idx]
    max_loc = max_locs[best_idx]
    
    scale = 0.5 * ((bottom_right[0] - top_left[0]) + (bottom_right[1] - top_left[1]))
    
    return scale, top_left, bottom_right

def compute_contour(mask,ord=30):
    
    contours, hierarchy = cv.findContours((mask).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = np.squeeze(contours[-1], axis=1)

    coeffs = (elliptic_fourier_descriptors(contour, order=ord))
    a0, c0 = calculate_dc_coefficients(contour)

    recon = reconstruct_contour(coeffs, locus=(a0,c0), num_points = len(contour))

    # with this contour we can compute the minimum and maximum x and y values to get a tight bounding box:
    x = recon[:,0]
    y = recon[:,1]
    x_min = x.min() - 25
    x_max = x.max() + 25
    y_min = y.min() - 25
    y_max = y.max() + 25
    box_bounds = np.array([x_min, y_min, x_max, y_max]).astype(int)
    
    return recon, box_bounds

def get_length_landmarks(recon):
    # here is where we start to find specific landmarks along the smooth approximation of the contour:

    # landmark 2: the nose of the fish should be the left-most point **after correcting the orientation**
    lm2_idx = np.argmin(recon[:,0])

    # now shift the contour so it begins at the nose of the fish and follows the contour ccw
    recon = np.concatenate([recon[lm2_idx:-1], recon[0:lm2_idx]],axis=0)
    x = recon[:,0]
    y = recon[:,1]

    # locate corners of caudal fin:
    x_max = x.max()
    caudal_peaks, _ = find_peaks(x, height = 0.8 * x_max)
    if len(caudal_peaks) == 1: 
        # we don't find the right-most corners of the caudal fin
        # so choose a point closest to average height of the back side of the fish
        valid_idxs = np.ndarray.flatten(np.argwhere(x > 0.75*x_max))
        valid_ys = y[valid_idxs]
        idx = np.argmin(np.abs(np.mean(valid_ys)-valid_ys))
        lm16_idx = valid_idxs[idx]
    # landmark 16: the indent in the middle of the caudal fin 
    # will be specified to be the minimum x-value betwen the caudal fin corners
    else:
        lm16_idx = caudal_peaks[0] + np.argmin(x[caudal_peaks[0]:caudal_peaks[1]])

    # the distance between landmark 2 and 16 is a good estimate of the length:

    # length = np.linalg.norm(recon[lm16_idx] - recon[lm2_idx])
    
    return recon, lm2_idx, lm16_idx

def rotate_image(image, angle, center_point=None):
    
    if len(image.shape) == 3:
        rows, cols, ch = image.shape
    else: 
        rows, cols = image.shape
    if center_point is None:
        image_center = tuple(np.array([cols, rows]) / 2)
        center_point = image_center

    rot_mat = cv.getRotationMatrix2D(center_point, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, (cols, rows), flags=cv.INTER_LINEAR)
    return result, rot_mat
  
# fish class, segments fish, fins, eyeball & computes a FL & (predicted) non-fin area
class fish():
    
    def __init__(self, image_path, predictor, write_mask = True, scale=None, num_fish=None):
        
        self.im_path = image_path
        image_path_split = os.path.split(self.im_path)
        self.dir = os.path.split(image_path_split[0])[1]
        self.im_name, self.ext = os.path.splitext(image_path_split[1])
        self.predictor = predictor
        self.scale=scale
        self.write_mask = write_mask
            
        # for images with multiple fish
        if num_fish is None:
            num_fish=1
        if num_fish > 1:
            idx = re.search(self.ext, self.im_path).start()
            new_im_path = self.im_path[:idx-2]+self.im_path[idx:]
            self.im_path = new_im_path
        
        # read in image and set initialize SAM
        self.image = cv.cvtColor(cv.imread(self.im_path), cv.COLOR_BGR2RGB)
        self.copy = np.copy(self.image)
        self.dims = np.shape(self.image[:,:,0])
        self.predictor.set_image(self.image)
        
        if os.path.exists('segmentations/'+self.dir):
            pass
        else:
            os.mkdir('segmentations/'+self.dir)
            
    def input_measurements(self):
        # TODO
        pass
    
    def get_measurments(self):
        # read in ROI and orienation data from csv, *or ask user for input* (*TODO)
        if os.path.exists('measurements/' + self.dir + '/' + self.im_name+'.csv'):

            write_mask=True
            with open('measurements/' + self.dir + '/' + self.im_name+'.csv', newline='') as csvfile:
                reader=csv.reader(csvfile, delimiter=',')
                for (j,row) in enumerate(reader):
                    if j==0:
                        crop_data = row

            roi_str = crop_data[0][1:-1].split()
            roi = [int(roi_str[0]), int(roi_str[1]), int(roi_str[2]), int(roi_str[3])]
            self.horiz_flip = crop_data[1]
            self.vertical_flip = crop_data[2]
            if len(crop_data) > 3:
                self.quality = crop_data[3]
            else:
                self.quality=None

            self.prediction_box = np.array([roi[0],roi[1],roi[0]+roi[2], roi[1]+roi[3]])
        else:
            pass #input_measurements(self)
        
    def get_scale(self):
        # predict scale based on template matching with fixed grid image
        if self.scale is None:
            self.scale, tl, br = get_grid_scale(self.image)

            self.scale = 5/self.scale
            
    def segment_fish(self):
        fish_masks,q,o = self.predictor.predict(box=self.prediction_box, multimask_output=True)
        idx=np.argmax(q)

        self.fish_mask = fish_masks[idx]
        self.fish_mask_full = self.fish_mask
        
        # flip as needed:
        if self.horiz_flip =='1':
            print('flipped horizontally')
            self.fish_mask=self.fish_mask[:,::-1]
            self.copy = self.copy[:, ::-1]
        if self.vertical_flip == '1':
            print('flipped vertically')
            self.fish_mask=self.fish_mask[::-1,:]
            self.copy = self.copy[::-1, :]
            
        self.recon, self.box_bounds = compute_contour(self.fish_mask, ord=40)#3
        self.fish_mask = self.fish_mask[self.box_bounds[1]:self.box_bounds[3],self.box_bounds[0]:self.box_bounds[2]]
        self.rgb_mask = np.stack([self.fish_mask, self.fish_mask, self.fish_mask], axis=-1)
        self.cropped_image = self.copy[self.box_bounds[1]:self.box_bounds[3],self.box_bounds[0]:self.box_bounds[2]]
        self.cropped_dims = np.shape(self.cropped_image[:,:,0])
        self.cropped_masked_image = self.copy[self.box_bounds[1]:self.box_bounds[3],self.box_bounds[0]:self.box_bounds[2]] * self.rgb_mask
        
        self.offset = np.array([self.box_bounds[0], self.box_bounds[1]])
        
        # check to see if the segmentation is degenerate
        if np.min(self.fish_mask.shape) < 50:
            self.degenerate=True
        else:
            self.degenerate=False

    def level_fish(self):
        
        x_min_idx = np.argmin(self.recon[:,0])
        x_max_idx = np.argmax(self.recon[:,0])

        recon_imag = np.zeros_like(self.fish_mask)
        recon_offset = self.recon - self.offset
        for idx in recon_offset.astype(int):
            recon_imag[idx[1],idx[0]]=1

        recon_imag[recon_offset[x_min_idx][1].astype(int)-5:recon_offset[x_min_idx][1].astype(int)+5, recon_offset[x_min_idx][0].astype(int)-5:recon_offset[x_min_idx][0].astype(int)+5] = 2
        recon_imag[recon_offset[x_max_idx][1].astype(int)-5:recon_offset[x_max_idx][1].astype(int)+5, recon_offset[x_max_idx][0].astype(int)-5:recon_offset[x_max_idx][0].astype(int)+5] = 2

        # cv.imwrite('fork_length_pts.jpg', recon_imag*128)

        fork_length_vector = self.recon[x_max_idx] - self.recon[x_min_idx]
        FL = np.linalg.norm(fork_length_vector) * self.scale
        self.area = np.sum(self.fish_mask)*(self.scale**2)
        print('for: ', self.im_path)
        print('fork length is: ', FL)
        print('area is: ', self.area)
        # print('no fin area is: ', self.no_fin_area)

                
        fork_length_dir = fork_length_vector/np.linalg.norm(fork_length_vector)
            
        self.fish_angle = np.arccos(np.dot(np.array([1,0]), fork_length_dir))*180/np.pi
        if fork_length_dir[1] < 0:
            self.fish_angle = -1 * self.fish_angle
            
        self.cp = 0.5*(recon_offset[x_min_idx] + recon_offset[x_max_idx]) # average

        cropped_masked_rotated_image, _ = rotate_image(self.cropped_masked_image, self.fish_angle, center_point=self.cp)

        if self.write_mask:
            cv.imwrite('cropped_masked_rotated.jpg', cropped_masked_rotated_image)
            # cv.imwrite('/Users/brknight/Documents/GitHub/HandsFreeFishing/segmentations/'+dir+'/rotated_mask_' + im_name, cropped_masked_rotated_image)


        self.rotated_mask, self.rot_mat = rotate_image(self.fish_mask.astype(np.uint8), self.fish_angle, center_point=self.cp)
        self.recon_offset_rotated =  (self.rot_mat[:2,:2]@(recon_offset.transpose())).transpose() + self.rot_mat[:,2]

        recon_imag_rot = np.zeros_like(self.fish_mask)
        for idx in self.recon_offset_rotated.astype(int):
            if (idx[1] < np.shape(recon_imag_rot)[0]) and (idx[0] < np.shape(recon_imag_rot)[1]):
                recon_imag_rot[idx[1],idx[0]]=1

        self.recon_offset_rotated, lm2_idx, lm16_idx = get_length_landmarks(self.recon_offset_rotated)

        recon_imag_rot[self.recon_offset_rotated[0][1].astype(int)-5:self.recon_offset_rotated[0][1].astype(int)+5, self.recon_offset_rotated[0][0].astype(int)-5:self.recon_offset_rotated[0][0].astype(int)+5] = 2
        recon_imag_rot[self.recon_offset_rotated[lm16_idx][1].astype(int)-5:self.recon_offset_rotated[lm16_idx][1].astype(int)+5, self.recon_offset_rotated[lm16_idx][0].astype(int)-5:self.recon_offset_rotated[lm16_idx][0].astype(int)+5] = 2

        fork_length_vector = self.recon_offset_rotated[lm16_idx] - self.recon_offset_rotated[0]
        self.FL = np.linalg.norm(fork_length_vector) * self.scale

        with open('measurements/' + self.dir + '/' + self.im_name+'.csv', 'a', newline='') as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow([self.scale, self.FL, self.area])
        
        if self.write_mask:
            cv.imwrite('mask_pred_rotated.jpg', self.rotated_mask*255)
            cv.imwrite('fork_length_pts_rotated.jpg', (recon_imag_rot*128))
            cv.imwrite('fork_length_pts_rotated2.jpg', (self.rotated_mask*128 + recon_imag_rot*128))
            
    def get_box(self, x_vals, y_vals):
        
        (m,n) = self.cropped_dims

        box = np.array([np.max([np.min(x_vals),0]), np.max([np.min(y_vals),0]), np.min([np.max(x_vals), n]), np.min([np.max(y_vals), m])])
        box = box + np.array([self.box_bounds[0], self.box_bounds[1], self.box_bounds[0], self.box_bounds[1]])
        
        if self.horiz_flip=="1":
            (m,n) = self.dims
            box = np.array([n - box[2], box[1], n - box[0], box[3]]) 
        
        if self.vertical_flip=="1":
            (m,n) = self.dims
            box = np.array([box[0], m - box[3], box[2], m - box[1]]) 
            
        return box     

    def get_eye_box(self, eye_ratio):
        x = self.recon_offset_rotated[:,0]
        y = self.recon_offset_rotated[:,1]
        x_right_side_idxs=np.argwhere(np.array([x < 0.25*np.mean(x), x > 0.1*np.mean(x)]).all(axis=0))
        mean_x = np.mean(x[x_right_side_idxs])
        eye_level = np.mean(y[x_right_side_idxs])
        
        mms = self.FL*eye_ratio # estimate what proportion of the fish length will contain the dorsal fin
        
        min_x = mean_x - mms/self.scale/2
        max_x = mean_x + mms/self.scale/2
        min_y = eye_level - mms/self.scale/2
        max_y = eye_level + mms/self.scale/2

        tl_rot = self.rot_mat[:2,:2].transpose()@(np.array([min_x,min_y]) - self.rot_mat[:,2])
        bl_rot = self.rot_mat[:2,:2].transpose()@(np.array([min_x,max_y]) - self.rot_mat[:,2])
        tr_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x,min_y]) - self.rot_mat[:,2])
        br_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x,max_y]) - self.rot_mat[:,2])
        corners = [tl_rot, tr_rot, br_rot, bl_rot]    
            
        x_vals = np.array([v[0] for v in corners])
        y_vals = np.array([v[1] for v in corners])
                    
        self.eye_box = self.get_box(x_vals, y_vals)

    def get_caudal_box(self, caudal_ratio):
        x = self.recon_offset_rotated[:,0]
        y = self.recon_offset_rotated[:,1]
        max_x = np.max(x)
        x_right_side_idxs=np.argwhere(x > np.mean(x))
        min_y = np.min(y[x_right_side_idxs])
        max_y = np.max(y[x_right_side_idxs])
        # caudal_box = np.array([max_x - 18/scale, min_y, max_x,  max_y])

        mms = self.FL*caudal_ratio # estimate what proportion of the fish length will contain the caudal fin

        tl_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x-mms/self.scale,min_y]) - self.rot_mat[:,2])
        bl_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x-mms/self.scale,max_y]) - self.rot_mat[:,2])
        tr_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x,max_y]) - self.rot_mat[:,2])
        br_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x,min_y]) - self.rot_mat[:,2])
        corners = [tl_rot, tr_rot, br_rot, bl_rot]    
            
        x_vals = np.array([v[0] for v in corners])
        y_vals = np.array([v[1] for v in corners])
        
        self.caudal_box = self.get_box(x_vals, y_vals)

    def get_dorsal_box(self, dorsal_ratio):
        x = self.recon_offset_rotated[:,0]
        y = self.recon_offset_rotated[:,1]
        x_middle_idxs=np.argwhere(np.array([x < 0.6*np.max(x), x > 0.5*np.max(x)]).all(axis=0)) # was x > 0.4*...
        min_y = np.min(y[x_middle_idxs])
        top_idx = np.argwhere(y==min_y)[0]
        top_x = x[top_idx]
        top_y = y[top_idx]
        
        mms = self.FL*dorsal_ratio # estimate what proportion of the fish length will contain the dorsal fin
        
        min_x = top_x[0] - mms/self.scale/2
        max_x = top_x[0] + mms/self.scale/2
        # max_y = top_y[0] + mms/self.scale/2
        # min_y = top_y[0] - mms/self.scale/10
        max_y = top_y[0] + mms/self.scale/3
        min_y = top_y[0] - mms/self.scale/3
            
        tl_rot = self.rot_mat[:2,:2].transpose()@(np.array([min_x,min_y]) - self.rot_mat[:,2])
        bl_rot = self.rot_mat[:2,:2].transpose()@(np.array([min_x,max_y]) - self.rot_mat[:,2])
        tr_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x,min_y]) - self.rot_mat[:,2])
        br_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x,max_y]) - self.rot_mat[:,2])
        corners = [tl_rot, tr_rot, br_rot, bl_rot]    
            
        x_vals = np.array([v[0] for v in corners])
        y_vals = np.array([v[1] for v in corners])
            
        self.dorsal_box = self.get_box(x_vals, y_vals)

    def get_pectoral_box(self, pectoral_ratio):
        x = self.recon_offset_rotated[:,0]
        y = self.recon_offset_rotated[:,1]
        x_left_middle_idxs=np.argwhere(np.array([x > 0.2*np.mean(x), x < 0.4*np.mean(x)]).all(axis=0))
        max_y = np.max(y[x_left_middle_idxs])
        pectoral_idx = np.argwhere(y==max_y)[0]
        pectoral_x = x[pectoral_idx]
        pectoral_y = y[pectoral_idx]
        
        mms = self.FL*pectoral_ratio # estimate what proportion of the fish length will contain the dorsal fin
        
        min_x = pectoral_x[0] - mms/self.scale/2
        max_x = pectoral_x[0] + mms/self.scale/2
        max_y = pectoral_y[0] + mms/self.scale/2
        min_y = pectoral_y[0] - mms/self.scale/2

        tl_rot = self.rot_mat[:2,:2].transpose()@(np.array([min_x,min_y]) - self.rot_mat[:,2])
        bl_rot = self.rot_mat[:2,:2].transpose()@(np.array([min_x,max_y]) - self.rot_mat[:,2])
        tr_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x,min_y]) - self.rot_mat[:,2])
        br_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x,max_y]) - self.rot_mat[:,2])
        corners = [tl_rot, tr_rot, br_rot, bl_rot]    
            
        x_vals = np.array([v[0] for v in corners])
        y_vals = np.array([v[1] for v in corners])
                    
        self.pectoral_box =  self.get_box(x_vals, y_vals)

    def get_pelvic_box(self, pectoral_ratio):
        x = self.recon_offset_rotated[:,0]
        y = self.recon_offset_rotated[:,1]
        x_left_middle_idxs=np.argwhere(np.array([x > 0.5*np.max(x), x < 0.6*np.max(x)]).all(axis=0)) # was x > 0.5*..
        max_y = np.max(y[x_left_middle_idxs])
        pelvic_idx = np.argwhere(y==max_y)[0]
        pelvic_x = x[pelvic_idx]
        pelvic_y = y[pelvic_idx]
        
        mms = self.FL*pectoral_ratio # estimate what proportion of the fish length will contain the dorsal fin
        
        min_x = pelvic_x[0] - mms/self.scale/2
        max_x = pelvic_x[0] + mms/self.scale/2
        max_y = pelvic_y[0] + mms/self.scale/10
        min_y = pelvic_y[0] - mms/self.scale/2

        tl_rot = self.rot_mat[:2,:2].transpose()@(np.array([min_x,min_y]) - self.rot_mat[:,2])
        bl_rot = self.rot_mat[:2,:2].transpose()@(np.array([min_x,max_y]) - self.rot_mat[:,2])
        tr_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x,min_y]) - self.rot_mat[:,2])
        br_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x,max_y]) - self.rot_mat[:,2])
        corners = [tl_rot, tr_rot, br_rot, bl_rot]    
            
        x_vals = np.array([v[0] for v in corners])
        y_vals = np.array([v[1] for v in corners])
                    
        self.pelvic_box = self.get_box(x_vals, y_vals)

    def get_anal_box(self, anal_ratio):
        x = self.recon_offset_rotated[:,0]
        y = self.recon_offset_rotated[:,1]
        x_right_middle_idxs=np.argwhere(np.array([x > 0.6*np.max(x), x < 0.7*np.max(x)]).all(axis=0))
        max_y = np.max(y[x_right_middle_idxs])
        anal_idx = np.argwhere(y==max_y)[0]
        anal_x = x[anal_idx]
        anal_y = y[anal_idx]
        
        mms = self.FL*anal_ratio # estimate what proportion of the fish length will contain the dorsal fin
        
        min_x = anal_x[0] - mms/(self.scale)/2
        max_x = anal_x[0] + mms/(self.scale)/2
        max_y = anal_y[0] + mms/(self.scale)/2
        min_y = anal_y[0] - mms/(self.scale)/2   
        
        tl_rot = self.rot_mat[:2,:2].transpose()@(np.array([min_x, min_y]) - self.rot_mat[:,2])
        bl_rot = self.rot_mat[:2,:2].transpose()@(np.array([min_x, max_y]) - self.rot_mat[:,2])
        tr_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x, min_y]) - self.rot_mat[:,2])
        br_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x, max_y]) - self.rot_mat[:,2])
        
        corners = [tl_rot, tr_rot, br_rot, bl_rot]    
            
        x_vals = np.array([v[0] for v in corners])
        y_vals = np.array([v[1] for v in corners])
                    
        self.anal_box = self.get_box(x_vals, y_vals)

    def get_adipose_box(self, adipose_ratio):
        x = self.recon_offset_rotated[:,0]
        y = self.recon_offset_rotated[:,1]
        x_right_middle_idxs=np.argwhere(np.array([x > 0.7*np.max(x), x < 0.8*np.max(x)]).all(axis=0))
        min_y = np.min(y[x_right_middle_idxs])
        anal_idx = np.argwhere(y==min_y)[0]
        anal_x = x[anal_idx]
        anal_y = y[anal_idx]
        
        mms = self.FL*adipose_ratio # estimate what proportion of the fish length will contain the dorsal fin
        
        min_x = anal_x[0] - mms/self.scale/2
        max_x = anal_x[0] + mms/self.scale/2
        max_y = anal_y[0] + mms/self.scale/4
        min_y = anal_y[0] - mms/self.scale/5

        tl_rot = self.rot_mat[:2,:2].transpose()@(np.array([min_x,min_y]) - self.rot_mat[:,2])
        bl_rot = self.rot_mat[:2,:2].transpose()@(np.array([min_x,max_y]) - self.rot_mat[:,2])
        tr_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x,min_y]) - self.rot_mat[:,2])
        br_rot = self.rot_mat[:2,:2].transpose()@(np.array([max_x,max_y]) - self.rot_mat[:,2])
        corners = [tl_rot, tr_rot, br_rot, bl_rot]    
            
        x_vals = np.array([v[0] for v in corners])
        y_vals = np.array([v[1] for v in corners])
                    
        self.adipose_box = self.get_box(x_vals, y_vals)

    def get_mask(self, box):
                    
        mask, q, o = self.predictor.predict(box=box)

        idx = np.argmax(q)
        best_mask = mask[idx].astype(np.uint8)
        # kernel = np.ones((15, 15), np.uint8) 
        # best_mask = cv.erode(best_mask, kernel,iterations=1)
        
        return best_mask# get_largest_connected_component(best_mask)
    
    def write_mask(self, mask, name="caudal"):
        
        if self.horiz_flip=="1":
            mask = mask[:,::-1]
        if self.vertical_flip=="1":
            mask = mask[::-1, :]
            
        cv.imwrite('segmentations/'+self.dir+'/'+name+'_mask_' + self.im_name + self.ext, 255*mask[self.box_bounds[1]:self.box_bounds[3],self.box_bounds[0]:self.box_bounds[2]])
            
    def get_full_segmentations(self):
        
        self.full_segmentation = self.fish_mask_full.astype(np.uint8)
        self.no_fin_segmentation = self.fish_mask_full.astype(np.uint8)
        
        masks = [self.eye_mask, self.dorsal_mask, self.adipose_mask, self.caudal_mask,
                 self.anal_mask, self.pelvic_mask, self.pectoral_mask]
        
        for (i,mask) in enumerate(masks):
            
            # creating a rough label image
            self.full_segmentation += (75*(i+1)) * mask
            
            if (i > 0 and i < 5): # don't zero out the eyeball, pelvic, or pectoral fins
                self.no_fin_segmentation = self.no_fin_segmentation * (mask ==0)
                
        # for additional contrast
        self.full_segmentation = -50*(self.full_segmentation == 0) + self.full_segmentation
        self.no_fin_segmentation = cv.erode(self.no_fin_segmentation, np.ones((5,5), np.uint8))
        # if self.write_mask:
        #     if self.horiz_flip=="1":
        #         self.full_segmentation = self.full_segmentation[:,::-1]
        #         self.no_fin_segmentation = self.no_fin_segmentation[:,::-1]
        #     if self.vertical_flip=="1":
        #         self.full_segmentation = self.full_segmentation[::-1, :]
        #         self.no_fin_segmentation = self.no_fin_segmentation[::-1, :]
                
        #     cv.imwrite('pred_all_mask.jpg', (self.full_segmentation[self.box_bounds[1]:self.box_bounds[3],self.box_bounds[0]:self.box_bounds[2]] + 50*self.fish_mask))
        #     cv.imwrite('pred_body_mask.jpg', 255*self.no_fin_segmentation[self.box_bounds[1]:self.box_bounds[3],self.box_bounds[0]:self.box_bounds[2]])

        #     cv.imwrite('/Users/brknight/Documents/GitHub/HandsFreeFishing/segmentations/'+self.dir+'/full_segementation_' + self.im_name, (50*self.fish_mask + self.full_segmentation[self.box_bounds[1]:self.box_bounds[3],self.box_bounds[0]:self.box_bounds[2]]))

    def get_fin_clips(self):
        
        self.get_eye_box(1/20)    
        self.get_dorsal_box(1/7)
        self.get_adipose_box(1/15)
        self.get_caudal_box(1/4.5)
        self.get_anal_box(1/10)
        self.get_pelvic_box(1/15)
        self.get_pectoral_box(1/15)
        
        self.eye_mask = self.get_mask(self.eye_box, )
        self.dorsal_mask = self.get_mask(self.dorsal_box)
        self.adipose_mask = self.get_mask(self.adipose_box)
        self.caudal_mask = self.get_mask(self.caudal_box)
        self.anal_mask = self.get_mask(self.anal_box)
        self.pelvic_mask = self.get_mask(self.pelvic_box)
        self.pectoral_mask = self.get_mask(self.pectoral_box)
    
    def get_no_fin_area(self):
        
        no_fin_mask = self.no_fin_segmentation
        if self.horiz_flip=="1":
            no_fin_mask = no_fin_mask[:,::-1]
        if self.vertical_flip=="1":
            no_fin_mask = no_fin_mask[::-1, :]
        box_slice = np.s_[self.box_bounds[1]:self.box_bounds[3], self.box_bounds[0]:self.box_bounds[2]]
        
        self.no_fin_area = np.sum(no_fin_mask[box_slice]) * self.scale**2

    def write_fin_masks(self):
            
        self.write_mask(self.eye_mask, name="eye")
        self.write_mask(self.dorsal_mask, name="dorsal")
        self.write_mask(self.adipose_mask, name="adipose")
        self.write_mask(self.caudal_mask, name="caudal")
        self.write_mask(self.anal_mask, name="anal")
        self.write_mask(self.pelvic_mask, name="pelvic")
        self.write_mask(self.pectoral_mask, name="pectoral")
    
    def write_full_masks(self):
        
        if os.path.exists('segmentations/'+self.dir):
            pass
        else:
            os.mkdir('segmentations/'+self.dir)
            
        cv.imwrite('segmentations/'+self.dir+'/'+'full_mask_' + self.im_name + self.ext, 10*self.full_segmentation)
        cv.imwrite('segmentations/'+self.dir+'/'+'no_fin_mask_' + self.im_name + self.ext, 10*self.no_fin_segmentation)
        
    def run(self):
        self.get_measurments()
        self.get_scale()
        self.segment_fish()
        self.level_fish()
        self.get_fin_clips()
        self.get_full_segmentations()
        self.get_no_fin_area()
        
def main():
    # example:
    image_path = 'sushi/example_fish/110524FishID6c.jpg'

    sam = sam_model_registry['vit_l'](checkpoint="./sam_vit_l_0b3195.pth")
    predictor = SamPredictor(sam)   
    
    myFish = fish(image_path, predictor,write_mask=True)
    myFish.run()
    box = myFish.prediction_box
    slice = np.s_[box[1]:box[3], box[0]:box[2]]

    print('\n\ntotal area = ', myFish.area, '\n\n')
    print('\n\ntotal non-fin area = ', myFish.no_fin_area, '\n\n')
    # Create a figure and a set of subplots
    save_figures = True # save figures
    
    fig, axes = plt.subplots(1, 3)

    # Display the first image on the left subplot
    axes[0].imshow(myFish.image[slice])
    axes[0].set_title('Raw Input')

    # Display the second image on the middle subplot
    axes[1].imshow(myFish.full_segmentation[slice])
    axes[1].set_title('Segmentation')
    
    # Display the third image on the right subplot
    axes[2].imshow(myFish.no_fin_segmentation[slice])
    axes[2].set_title('No Fin Segmentation')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    if save_figures:
        plt.savefig('fin_clipping_example.eps',dpi=200)
        plt.savefig('fin_clipping_example.png',dpi=200)

    # Show the plot
    plt.show()
    
if __name__ == '__main__':
    main()
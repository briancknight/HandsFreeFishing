import numpy as np
import os
import csv
from tifffile import imread, imwrite
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
import cv2 as cv
from pyefd import elliptic_fourier_descriptors, calculate_dc_coefficients, plot_efd, reconstruct_contour
import pandas as pd
import regex as re
from skimage.morphology import convex_hull_image
from ellipse import LsqEllipse

# in order to read the template image
from importlib.resources import files

def load_template(package, resource):
    image_path = files('HandsFreeFishing').joinpath('IMG_2937_grid_template.tif')
    return image_path
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

def get_grid_scale(img,ds=2):
    """Computes the relative scale of the image by finding the best match for a template image of a square grid of various sizes

    Args:
        img (_type_): _description_

    Returns:
        scale (int): the number of pixels per grid square in the given image
        top_left (np.ndarray): coordinates of top left corner of best match
        bottom_right (np.ndarray): coordinates of bottom right corner of best match
    """
    img = cv.cvtColor(img[::ds,::ds], cv.COLOR_RGB2GRAY)
    template = cv.imread(load_template('templates', 'IMG_2937_grid_template.tif'), cv.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]
    
    meth = 'TM_CCOEFF_NORMED'
    sizes = np.linspace(0.25, 4, 150)
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
    bottom_right = np.array(brs[best_idx])*ds
    top_left = np.array(tls[best_idx])*ds
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
    
    def __init__(self, image_path, predictor, write_masks = True, mask_ext = '.png', scale=None, num_fish=None):
        
        self.im_path = image_path
        image_path_split = os.path.split(self.im_path)
        self.dir = os.path.split(image_path_split[0])[1]
        self.im_name, self.ext = os.path.splitext(image_path_split[1])
        self.mask_ext = mask_ext
        self.predictor = predictor
        self.scale=scale
        self.write_masks = write_masks
        self.n_steps=2
        self.ord=100
            
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
        
        if os.path.exists(os.path.join('segmentations',self.dir)):
            pass
        else:
            os.makedirs(os.path.join('segmentations',self.dir),exist_ok=True)
            
    def input_measurements(self):
        # TODO
        pass
    
    def get_measurements(self):
        # read in ROI and orienation data from csv, *or ask user for input* (*TODO)
        measurement_path = os.path.join('measurements', self.dir, self.im_name+'.csv')
        if os.path.exists(measurement_path):

            write_masks=True
            with open(measurement_path, newline='') as csvfile:
                reader=csv.reader(csvfile, delimiter=',')
                crop_data=next(reader)
                # for (j,row) in enumerate(reader):
                #     if j==0:
                #         crop_data = row
                #     break

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
        
    def get_scale(self,ds=1):
        # predict scale based on template matching with fixed grid image
        
        if self.frozen:
            measurement_path = os.path.join('measurements', self.dir, self.im_name+'.csv')
            with open(measurement_path, newline='') as csvfile:
                reader=csv.reader(csvfile, delimiter=',')
                for (j,row) in enumerate(reader):
                    if j==1:
                        scale_data = row
            self.scale = float(scale_data[0])
            
        if self.scale is None:
            self.scale, self.grid_tl, self.grid_br = get_grid_scale(self.image,ds=ds)

            self.scale = 5/self.scale
            
    def segment_fish(self):
        
        fish_mask_path = os.path.join('segmentations', self.dir, 'initial_mask_' + self.im_name + self.mask_ext)
        
        if os.path.exists(fish_mask_path):
            print('using existing segmentation at: ', fish_mask_path)
            self.fish_mask = cv.imread(fish_mask_path)[:,:,0] == 255
        else:
            self.predictor.set_image(self.image)
            fish_masks,q,o = self.predictor.predict(box=self.prediction_box, multimask_output=True)
            idx=np.argmax(q)
            self.fish_mask = fish_masks[idx]
        
        self.fish_mask =  get_largest_connected_component((self.fish_mask*255).astype(np.uint8))*255
        self.fish_mask_full = np.copy(self.fish_mask)
                
        # flip as needed:
        if self.horiz_flip =='1':
            print('flipped horizontally')
            self.fish_mask=self.fish_mask[:,::-1]
            self.copy = self.copy[:, ::-1]
        if self.vertical_flip == '1':
            print('flipped vertically')
            self.fish_mask=self.fish_mask[::-1,:]
            self.copy = self.copy[::-1, :]
            
        self.recon, self.box_bounds = compute_contour(self.fish_mask, ord=self.ord)#3
        self.fish_mask = self.fish_mask[self.box_bounds[1]:self.box_bounds[3],self.box_bounds[0]:self.box_bounds[2]]
        self.rgb_mask = np.stack([self.fish_mask, self.fish_mask, self.fish_mask], axis=-1)
        self.cropped_image = self.copy[self.box_bounds[1]:self.box_bounds[3],self.box_bounds[0]:self.box_bounds[2]]
        self.cropped_dims = np.shape(self.cropped_image[:,:,0])
        self.cropped_masked_image = self.copy[self.box_bounds[1]:self.box_bounds[3],self.box_bounds[0]:self.box_bounds[2]] * self.rgb_mask
        
        self.offset = np.array([self.box_bounds[0], self.box_bounds[1]])
        
        # check to see if the segmentation is degenerate
        print('max dimension size (mm): ', np.max(self.fish_mask.shape)*self.scale)
        print(np.max(self.fish_mask.shape))
        if np.max(self.fish_mask.shape)*self.scale < 8:
            self.degenerate=True
            cv.imwrite(os.path.join('segmentations',self.dir,'degenerate_' + self.im_name + '.png'), self.fish_mask_full*255)
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
        # print('no fin area is: ', self.no_fin_area)

                
        fork_length_dir = fork_length_vector/np.linalg.norm(fork_length_vector)
            
        self.fish_angle = np.arccos(np.dot(np.array([1,0]), fork_length_dir))*180/np.pi
        if fork_length_dir[1] < 0:
            self.fish_angle = -1 * self.fish_angle
            
        self.cp = 0.5*(recon_offset[x_min_idx] + recon_offset[x_max_idx]) # average

        cropped_masked_rotated_image, _ = rotate_image(self.cropped_masked_image, self.fish_angle, center_point=self.cp)

        if self.write_masks:
            cv.imwrite('cropped_masked_rotated.jpg', cropped_masked_rotated_image)
            # cv.imwrite('/Users/brknight/Documents/GitHub/HandsFreeFishing/segmentations/'+dir+'/rotated_mask_' + im_name, cropped_masked_rotated_image)


        self.rotated_mask, self.rot_mat = rotate_image(self.fish_mask.astype(np.uint8), self.fish_angle, center_point=self.cp)
        self.recon_offset_rotated =  (self.rot_mat[:2,:2]@(recon_offset.transpose())).transpose() + self.rot_mat[:,2]

        recon_imag_rot = np.zeros_like(self.fish_mask)
        for idx in self.recon_offset_rotated.astype(int):
            if (idx[1] < np.shape(recon_imag_rot)[0]) and (idx[0] < np.shape(recon_imag_rot)[1]):
                recon_imag_rot[idx[1],idx[0]]=1

        self.recon_offset_rotated, lm2_idx, lm16_idx = get_length_landmarks(self.recon_offset_rotated)
        self.FL_points=(self.rot_mat[:2,:2].transpose()@(self.recon_offset_rotated[lm16_idx] - self.rot_mat[:,2]) + self.offset, 
                        self.rot_mat[:2,:2].transpose()@(self.recon_offset_rotated[0] - self.rot_mat[:,2]) + self.offset)

        recon_imag_rot[self.recon_offset_rotated[0][1].astype(int)-5:self.recon_offset_rotated[0][1].astype(int)+5, self.recon_offset_rotated[0][0].astype(int)-5:self.recon_offset_rotated[0][0].astype(int)+5] = 2
        recon_imag_rot[self.recon_offset_rotated[lm16_idx][1].astype(int)-5:self.recon_offset_rotated[lm16_idx][1].astype(int)+5, self.recon_offset_rotated[lm16_idx][0].astype(int)-5:self.recon_offset_rotated[lm16_idx][0].astype(int)+5] = 2

        fork_length_vector = self.recon_offset_rotated[lm16_idx] - self.recon_offset_rotated[0]
        self.FL = np.linalg.norm(fork_length_vector) * self.scale

        print('for: ', self.im_path)
        print('fork length is: ', self.FL)
        print('area is: ', self.area)
        
        with open(os.path.join('measurements', self.dir, self.im_name+'.csv'), 'a', newline='') as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow([self.scale, self.FL, self.area])
        
        if self.write_masks:
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
        
        return get_largest_connected_component(best_mask)*255
    
    def write_mask(self, mask, name="caudal"):
        

        cv.imwrite(os.path.join('segmentations', self.dir, name+'_mask_' + self.im_name + self.mask_ext), 255*mask)

        # cv.imwrite(os.path.join('segmentations', self.dir, name+'_mask_' + self.im_name + self.mask_ext), 255*mask[self.prediction_box[1]:self.prediction_box[3],self.prediction_box[0]:self.prediction_box[2]])

        # if self.frozen:
        #     (name, "_mask is already written!")
        # else:
        #     # cv.imwrite(os.path.join('segmentations', self.dir, name+'_mask_' + self.im_name + self.mask_ext), 255*mask[self.box_bounds[1]:self.box_bounds[3],self.box_bounds[0]:self.box_bounds[2]])
        #     cv.imwrite(os.path.join('segmentations', self.dir, name+'_mask_' + self.im_name + self.mask_ext), 255*mask[self.prediction_box[1]:self.prediction_box[3],self.prediction_box[0]:self.prediction_box[2]])
            
    def get_full_segmentations(self):
        self.full_segmentation = np.stack([self.fish_mask_full, np.zeros_like(self.fish_mask_full), np.zeros_like(self.fish_mask_full)], axis=-1).astype(np.uint8)
        # self.full_segmentation = self.fish_mask_full
        self.no_fin_segmentation = self.fish_mask_full
        
        masks = [self.eye_mask, self.dorsal_mask, self.adipose_mask, self.caudal_mask,
                 self.anal_mask, self.pelvic_mask, self.pectoral_mask]
        
        for (i,mask) in enumerate(masks):
            
            # creating a rough label image
            demask = (mask==0)
            self.full_segmentation *= np.stack([demask, demask, demask], axis=-1)
            self.full_segmentation[:,:,1] +=  mask
            
            # if (i > 0 and i < 5): # don't zero out the eyeball, pelvic, or pectoral fins
            if (i > 0): # don't zero out the eyeball
                self.no_fin_segmentation = self.no_fin_segmentation * (mask ==0)
                
        # for additional contrast
        # self.full_segmentation = -50*(self.full_segmentation == 0) + self.full_segmentation
        self.no_fin_segmentation = cv.erode(self.no_fin_segmentation*1.0, np.ones((5,5), np.uint8), iterations=3) * 255
        self.no_fin_segmentation = get_largest_connected_component(((self.no_fin_segmentation>0) * 255).astype(np.uint8)) * 255
        # self.no_fin_segmentation = np.stack([self.no_fin_segmentation, self.no_fin_segmentation, self.no_fin_segmentation], axis=-1)
        
        self.full_segmentation = (self.full_segmentation>0) * 255
    
    def filet_fish(self,n_steps=6, ord=10):
    
        nf_mask = self.no_fin_segmentation.copy()
        if self.horiz_flip =='1':
            print('flipped horizontally')
            nf_mask = nf_mask[:,::-1]
        if self.vertical_flip == '1':
            print('flipped vertically')
            nf_mask = nf_mask[::-1,:]
        
        nf_mask_temp = convex_hull_image(nf_mask)
        eps=1e-4
        if np.sum(nf_mask_temp > 0)*(self.scale**2)/(self.no_fin_area+eps) > 1.25:
            print('too much')
        else:
            nf_mask = nf_mask_temp
        
        self.nf_recon, self.nf_box_bounds = compute_contour(nf_mask, ord=ord)#3
        self.nf_offset = np.array([self.nf_box_bounds[0], self.nf_box_bounds[1]])

        x_min_idx = np.argmin(self.nf_recon[:,0])
        x_max_idx = np.argmax(self.nf_recon[:,0])

        recon_offset = self.nf_recon - self.nf_offset
        
        fork_length_vector = self.nf_recon[x_max_idx] - self.nf_recon[x_min_idx]

        fork_length_dir = fork_length_vector/np.linalg.norm(fork_length_vector)
            
        self.nf_fish_angle = np.arccos(np.dot(np.array([1,0]), fork_length_dir))*180/np.pi
        
        if fork_length_dir[1] < 0:
            self.nf_fish_angle = -1 * self.nf_fish_angle
            
        self.nf_cp = 0.5*(recon_offset[x_min_idx] + recon_offset[x_max_idx]) # average
        rot_mat = cv.getRotationMatrix2D(self.nf_cp, self.nf_fish_angle, 1.0)

        recon_offset_rotated =  (rot_mat[:2,:2]@(recon_offset.transpose())).transpose() + rot_mat[:,2]

        x_min_idx = np.argmin(recon_offset_rotated[:,0])
        x_max_idx = np.argmax(recon_offset_rotated[:,0])
        recon_offset_rotated = np.roll(recon_offset_rotated, -x_min_idx, axis=0)
        x_min_idx = np.argmin(recon_offset_rotated[:,0])
        x_max_idx = np.argmax(recon_offset_rotated[:,0])
        
        recon_offset = (rot_mat[:2,:2].transpose()@((recon_offset_rotated - rot_mat[:,2]).transpose())).transpose()
        center_of_mass = np.mean(recon_offset, axis=0) + self.nf_offset
        
        top_step = int(np.ceil(x_max_idx/(n_steps)))
        bottom_step = int(np.ceil((len(recon_offset) - x_max_idx)/n_steps))
        top_steps = np.s_[0:x_max_idx:top_step]
        bottom_steps = np.s_[x_max_idx:len(recon_offset):bottom_step]
        
        steps = np.r_[top_steps, bottom_steps]
        sector_pts = recon_offset[steps].astype(int) + self.nf_offset
        
        self.filled_sectors=[]
        self.sector_areas = []
        
        lines = [np.linspace(sector_pts[n], center_of_mass, 150) for n in range(len(steps))]
        self.line_lengths = [np.linalg.norm(line[-1]-line[0])*self.scale for line in lines]
        
        for n in range(len(steps)):
            recon_sector_im = np.zeros_like(self.fish_mask_full)
            
            if n == len(steps)-1:
                sector = recon_offset[steps[n]:] + self.nf_offset
                contour = np.concatenate([sector, lines[n],lines[0]])
            else:
                sector = recon_offset[steps[n]:steps[n+1]] + self.nf_offset
                contour = np.concatenate([sector, lines[n],lines[n+1]])
                
            seed = np.mean(contour,axis=0).astype(int)
            width=3
            for idx in contour.astype(int):
                recon_sector_im[idx[1]-width:idx[1]+width,idx[0]-width:idx[0]+width] = 255

            h, w = recon_sector_im.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            sector_floodfill = recon_sector_im.astype(np.uint8)
            cv.floodFill(sector_floodfill, mask, seed,255)
            
            if self.horiz_flip =='1':
                sector_floodfill = sector_floodfill[:,::-1]
            if self.vertical_flip == '1':
                sector_floodfill = sector_floodfill[::-1,:]
            
            kernel = np.ones((3, 3), np.uint8)
            sector_floodfill = cv.erode(sector_floodfill,kernel,iterations=2)
            self.filled_sectors.append(sector_floodfill)
            # self.filled_sectors.append(sector_floodfill)
            self.sector_areas.append(np.sum(sector_floodfill > 0)*(self.scale**2))
        
    def get_partitioned_surface_area(self,n_partitions=6, ord=50):
    
        nf_mask = self.no_fin_segmentation.copy()
        if self.horiz_flip =='1':
            print('flipped horizontally')
            nf_mask = nf_mask[:,::-1]
        if self.vertical_flip == '1':
            print('flipped vertically')
            nf_mask = nf_mask[::-1,:]
        
        nf_mask_temp = convex_hull_image(nf_mask)
        eps=1e-4
        if np.sum(nf_mask_temp > 0)*(self.scale**2)/(self.no_fin_area+eps) > 1.25:
            print('too much')
        else:
            nf_mask = nf_mask_temp
        
        self.nf_recon, self.nf_box_bounds = compute_contour(nf_mask, ord=ord)#3
        self.nf_offset = np.array([self.nf_box_bounds[0], self.nf_box_bounds[1]])

        reg = LsqEllipse().fit(self.nf_recon)
        center, self.major_axis, self.minor_axis, phi = reg.as_parameters()
        self.major_axis *= self.scale
        self.minor_axis *= self.scale
                
        x_min_idx = np.argmin(self.nf_recon[:,0])
        x_max_idx = np.argmax(self.nf_recon[:,0])

        recon_offset = self.nf_recon - self.nf_offset
        
        fork_length_vector = self.nf_recon[x_max_idx] - self.nf_recon[x_min_idx]

        fork_length_dir = fork_length_vector/np.linalg.norm(fork_length_vector)
            
        self.nf_fish_angle = np.arccos(np.dot(np.array([1,0]), fork_length_dir))*180/np.pi
        
        if fork_length_dir[1] < 0:
            self.nf_fish_angle = -1 * self.nf_fish_angle
            
        self.nf_cp = 0.5*(recon_offset[x_min_idx] + recon_offset[x_max_idx]) # average
        rot_mat = cv.getRotationMatrix2D(self.nf_cp, self.nf_fish_angle, 1.0)

        recon_offset_rotated =  (rot_mat[:2,:2]@(recon_offset.transpose())).transpose() + rot_mat[:,2]

        x_min_idx = np.argmin(recon_offset_rotated[:,0])
        x_max_idx = np.argmax(recon_offset_rotated[:,0])
        recon_offset_rotated = np.roll(recon_offset_rotated, -x_min_idx, axis=0)
        x_min_idx = np.argmin(recon_offset_rotated[:,0])
        x_max_idx = np.argmax(recon_offset_rotated[:,0])
        
        recon_offset = (rot_mat[:2,:2].transpose()@((recon_offset_rotated - rot_mat[:,2]).transpose())).transpose()
        
        top_step = int(np.ceil(x_max_idx/(n_partitions)))
        bottom_step = int(np.ceil((len(recon_offset) - x_max_idx)/n_partitions))
        top_steps = np.s_[0:x_max_idx:top_step]
        bottom_steps = np.s_[x_max_idx:len(recon_offset):bottom_step]
        
        steps = np.r_[top_steps, bottom_steps]
        sector_pts = recon_offset[steps].astype(int) + self.nf_offset
        
        self.filled_partitions=[]
        self.partitioned_areas = []
        
        lines = [np.linspace(sector_pts[n], sector_pts[-n], 150) for n in range(1, n_partitions)]
        self.partition_line_lengths = [np.linalg.norm(line[-1]-line[0])*self.scale for line in lines]
        
        for n in range(n_partitions):
            recon_sector_im = np.zeros_like(self.fish_mask_full)
            
            if n == 0:
                sector = np.concatenate([recon_offset[steps[-1]:], recon_offset[0:steps[1]]]) + self.nf_offset
                contour = np.concatenate([sector, lines[0]])
            elif n == n_partitions-1:
                sector = recon_offset[steps[n_partitions-1]:steps[n_partitions+1]] + self.nf_offset
                contour = np.concatenate([lines[-1], sector])
            else:
                sector1 = recon_offset[steps[n]:steps[n+1]] + self.nf_offset
                sector2 = recon_offset[steps[-n-1]:steps[-n]] + self.nf_offset
                contour = np.concatenate([lines[n-1], sector1, sector2, lines[n]])
                
            seed = np.mean(contour,axis=0).astype(int)
            width=3
            for idx in contour.astype(int):
                recon_sector_im[idx[1]-width:idx[1]+width,idx[0]-width:idx[0]+width] = 255

            h, w = recon_sector_im.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            sector_floodfill = recon_sector_im.astype(np.uint8)
            cv.floodFill(sector_floodfill, mask, seed,255)
            
            if self.horiz_flip =='1':
                sector_floodfill = sector_floodfill[:,::-1]
            if self.vertical_flip == '1':
                sector_floodfill = sector_floodfill[::-1,:]
            
            kernel = np.ones((3, 3), np.uint8)
            sector_floodfill = cv.erode(sector_floodfill,kernel,iterations=2)
            self.filled_partitions.append(sector_floodfill)
            # self.filled_sectors.append(sector_floodfill)
            self.partitioned_areas.append(np.sum(sector_floodfill > 0)*(self.scale**2))
        
    def get_digitized_landmarks(self,n_steps=7, ord=100):
    
        nf_mask = self.no_fin_segmentation.copy()
        eye_mask = self.eye_mask.copy()
        if self.horiz_flip =='1':
            print('flipped horizontally')
            nf_mask = nf_mask[:,::-1]
            eye_mask = eye_mask[:,::-1]
        if self.vertical_flip == '1':
            print('flipped vertically')
            nf_mask = nf_mask[::-1,:]
            eye_mask=eye_mask[::-1,:]
        
        nf_mask_temp = convex_hull_image(nf_mask)
        eps=1e-4
        if np.sum(nf_mask_temp > 0)*(self.scale**2)/(self.no_fin_area+eps) > 1.25:
            print('too much')
        else:
            nf_mask = nf_mask_temp
        
        nf_recon, nf_box_bounds = compute_contour(nf_mask, ord=ord)#3
        eye_contour,_ = compute_contour(eye_mask, ord=ord)
        nf_offset = np.array([nf_box_bounds[0], nf_box_bounds[1]])

        x_min_idx = np.argmin(nf_recon[:,0])
        x_max_idx = np.argmax(nf_recon[:,0])

        recon_offset = nf_recon - nf_offset
        eye_contour_offset = eye_contour - nf_offset
        fork_length_vector = nf_recon[x_max_idx] - nf_recon[x_min_idx]

        fork_length_dir = fork_length_vector/np.linalg.norm(fork_length_vector)
            
        nf_fish_angle = np.arccos(np.dot(np.array([1,0]), fork_length_dir))*180/np.pi
        
        if fork_length_dir[1] < 0:
            nf_fish_angle = -1 * nf_fish_angle
            
        nf_cp = 0.5*(recon_offset[x_min_idx] + recon_offset[x_max_idx]) # average
        rot_mat = cv.getRotationMatrix2D(nf_cp, nf_fish_angle, 1.0)

        recon_offset_rotated =  (rot_mat[:2,:2]@(recon_offset.transpose())).transpose() + rot_mat[:,2]
        x_min_idx = np.argmin(recon_offset_rotated[:,0])
        x_max_idx = np.argmax(recon_offset_rotated[:,0])
        recon_offset_rotated = np.roll(recon_offset_rotated, -x_min_idx, axis=0)
        x_min_idx = np.argmin(recon_offset_rotated[:,0])
        x_max_idx = np.argmax(recon_offset_rotated[:,0])
        
        # rotate the eye contour by the same transformation
        eye_contour_rotated = (rot_mat[:2,:2]@(eye_contour_offset.transpose())).transpose() + rot_mat[:,2]
        x_max_idx_eye = np.argmax(eye_contour_rotated[:,0])
        
        # redefine recon_offset by unrotated shifted recon_offset_rotated, now with idx 0 corresponding to the minimum x index
        recon_offset = (rot_mat[:2,:2].transpose()@((recon_offset_rotated - rot_mat[:,2]).transpose())).transpose()
        # center_of_mass = np.mean(recon_offset, axis=0) + self.nf_offset
        
        top_step = int(np.ceil(x_max_idx/(n_steps)))
        bottom_step = int(np.ceil((len(recon_offset) - x_max_idx)/n_steps))
        top_steps = np.s_[0:x_max_idx:top_step]
        bottom_steps = np.s_[x_max_idx:len(recon_offset):bottom_step]
        
        steps = np.r_[top_steps, bottom_steps]
        sector_pts = recon_offset[steps].astype(int) + nf_offset
        eye_pt = eye_contour[x_max_idx_eye].astype(int)
        
        truss_pairs = [(0,1), (0,2), (0,-1), (0,-2), (0, 7), (0, 'eye'),
                 (1, 2),(1, 'eye'), (1, -1), (1, -2),
                 (2,3), (2,-1), (2,-2), (2,-3),
                 (3,4), (3,-2),(3,-3),(3,-4),
                 (4,5), (4, -3), (4, -4),
                 (5,6), (5,-4), (5,-5),(5, -6), 
                 (6,7), (6, -5), (6,-6),
                 (7,-6),
                 (-6, -5), 
                 (-5,-4),
                 (-4,-3),
                 (-3,-2),
                 (-2,-1)
                 ]
        # idxs = [6, 5, 8, 12, 13, 14, ]
        best_truss_pairs = [(0,'eye'), (0, 7), 
                      (1, 'eye'),
                      (2, -1), (2,-2),(2,-3),
                      (3,-2),(3,-4),
                      (4,5), (4,-3), (4,-4), 
                      (5, -6),
                      (6,7), (6, -5), 
                      (7,-6),
                      (-5,-4)
                      ]
        self.truss_points=[sector_pts[i] for i in range(len(sector_pts))]
        self.truss_points.append(eye_pt)
        
        self.truss_lengths = []
        self.best_truss_lengths = []
        self.truss_start_end_pts = []

        
        for (i,pair) in enumerate(truss_pairs):
            idx1 = pair[0]
            idx2 = pair[1]
            if type(idx2)==int:
                self.truss_start_end_pts.append([sector_pts[-idx1], sector_pts[-idx2]])
                self.truss_lengths.append(np.linalg.norm(sector_pts[-idx1] - sector_pts[-idx2]))
            else:
                if idx2=='eye':
                    self.truss_start_end_pts.append([sector_pts[-idx1], eye_pt])
                    self.truss_lengths.append(np.linalg.norm(sector_pts[-idx1] - eye_pt))
            
            if pair in best_truss_pairs:
                if type(idx2)==int:
                    self.best_truss_lengths.append(np.linalg.norm(sector_pts[-idx1] - sector_pts[-idx2]))
                else:
                    if idx2=='eye':
                        self.best_truss_lengths.append(np.linalg.norm(sector_pts[-idx1] - eye_pt))
                
    def get_fin_clips(self):
        
        self.get_eye_box(1/20)    
        self.get_dorsal_box(1/7)
        self.get_adipose_box(1/15)
        self.get_caudal_box(1/4.5)
        self.get_anal_box(1/10)
        self.get_pelvic_box(1/15)
        self.get_pectoral_box(1/15)
        
        self.eye_mask = self.get_mask(self.eye_box)
        self.dorsal_mask = self.get_mask(self.dorsal_box)
        self.adipose_mask = self.get_mask(self.adipose_box)
        self.caudal_mask = self.get_mask(self.caudal_box)
        self.anal_mask = self.get_mask(self.anal_box)
        self.pelvic_mask = self.get_mask(self.pelvic_box)
        self.pectoral_mask = self.get_mask(self.pectoral_box)
    
    def get_eye_diameter(self):
        
        self.eye_contour, _  = compute_contour(self.eye_mask, ord=self.ord)#3
        l = len(self.eye_contour)
        max_diameter = np.max([np.linalg.norm(self.eye_contour[i] - self.eye_contour[j]) for i in range(l) for j in range(l)])
        
        self.eye_diameter=self.scale*max_diameter
        
    def get_no_fin_area(self, convex_hull_correction=True):
        
        no_fin_mask = self.no_fin_segmentation.copy()
        box_slice = np.s_[self.box_bounds[1]:self.box_bounds[3], self.box_bounds[0]:self.box_bounds[2]]
        
        if self.horiz_flip=="1":
            no_fin_mask = no_fin_mask[:,::-1]
        if self.vertical_flip=="1":
            no_fin_mask = no_fin_mask[::-1, :]
        
        self.no_fin_area = np.sum(no_fin_mask[box_slice]>0) * self.scale**2
        print(self.no_fin_area)
        eps=1e-3
        
        if convex_hull_correction:
            no_fin_mask_temp = convex_hull_image(no_fin_mask.copy())
            no_fin_area_temp = np.sum(no_fin_mask_temp > 0)*(self.scale**2)
            print(no_fin_area_temp/self.no_fin_area)
            
            if no_fin_area_temp/(self.no_fin_area+eps) > 1.2:
                write_ch=False
                print('too much')
            else:
                write_ch=True
                no_fin_mask = no_fin_mask_temp
                
                if self.horiz_flip=="1":
                    no_fin_mask = no_fin_mask[:,::-1]
                if self.vertical_flip=="1":
                    no_fin_mask = no_fin_mask[::-1, :]
            
                self.no_fin_area = no_fin_area_temp
        
        print(self.no_fin_area)
        if write_ch:
            print('writing convex hull, in theory')
            self.write_mask(no_fin_mask, name='convex_hull')

    def write_fin_masks(self):
            
        self.write_mask(self.eye_mask, name="eye")
        self.write_mask(self.dorsal_mask, name="dorsal")
        self.write_mask(self.adipose_mask, name="adipose")
        self.write_mask(self.caudal_mask, name="caudal")
        self.write_mask(self.anal_mask, name="anal")
        self.write_mask(self.pelvic_mask, name="pelvic")
        self.write_mask(self.pectoral_mask, name="pectoral")
    
    def write_full_masks(self):
        if self.degenerate:
            print('write_full_masks failed due to degenerate segmentation')
        
        else:
            if os.path.exists(os.path.join('segmentations', self.dir)):
                pass
            else:
                os.makedirs(os.path.join('segmentations', self.dir),exist_ok=True)
                
            cv.imwrite(os.path.join('segmentations', self.dir, 'initial_mask_' + self.im_name + self.mask_ext), 255*self.fish_mask_full)  
            cv.imwrite(os.path.join('segmentations', self.dir, 'full_mask_' + self.im_name + self.mask_ext), 255*self.full_segmentation)
            cv.imwrite(os.path.join('segmentations', self.dir, 'no_fin_mask_' + self.im_name + self.mask_ext), 255*self.no_fin_segmentation)
    
    def check_freezer(self):
        self.frozen=False
        nf_seg_path = os.path.join('segmentations', self.dir, 'no_fin_mask_' + self.im_name + self.mask_ext)
        full_seg_path = os.path.join('segmentations', self.dir, 'full_mask_' + self.im_name + self.mask_ext)
        dorsal_seg_path = os.path.join('segmentations', self.dir, 'dorsal_mask_' + self.im_name + self.mask_ext)
        adipose_seg_path = os.path.join('segmentations', self.dir, 'adipose_mask_' + self.im_name + self.mask_ext)
        caudal_seg_path = os.path.join('segmentations', self.dir, 'caudal_mask_' + self.im_name + self.mask_ext)
        anal_seg_path = os.path.join('segmentations', self.dir, 'anal_mask_' + self.im_name + self.mask_ext)
        pelvic_seg_path = os.path.join('segmentations', self.dir, 'pelvic_mask_' + self.im_name + self.mask_ext)
        pectoral_seg_path = os.path.join('segmentations', self.dir, 'pectoral_mask_' + self.im_name + self.mask_ext)
        eye_seg_path = os.path.join('segmentations', self.dir, 'eye_mask_' + self.im_name + self.mask_ext)
        
        all_paths = [nf_seg_path, full_seg_path, dorsal_seg_path, adipose_seg_path, 
                     caudal_seg_path, anal_seg_path, pelvic_seg_path, pectoral_seg_path, 
                     eye_seg_path]
        
        if np.all([os.path.exists(path) for path in all_paths]):
            self.frozen=True

        print('frozen? ', self.frozen)
        
    def thaw(self):
        nf_seg_path = os.path.join('segmentations', self.dir, 'no_fin_mask_' + self.im_name + self.mask_ext)
        full_seg_path = os.path.join('segmentations', self.dir, 'full_mask_' + self.im_name + self.mask_ext)
        dorsal_seg_path = os.path.join('segmentations', self.dir, 'dorsal_mask_' + self.im_name + self.mask_ext)
        adipose_seg_path = os.path.join('segmentations', self.dir, 'adipose_mask_' + self.im_name + self.mask_ext)
        caudal_seg_path = os.path.join('segmentations', self.dir, 'caudal_mask_' + self.im_name + self.mask_ext)
        anal_seg_path = os.path.join('segmentations', self.dir, 'anal_mask_' + self.im_name + self.mask_ext)
        pelvic_seg_path = os.path.join('segmentations', self.dir, 'pelvic_mask_' + self.im_name + self.mask_ext)
        pectoral_seg_path = os.path.join('segmentations', self.dir, 'pectoral_mask_' + self.im_name + self.mask_ext)
        eye_seg_path = os.path.join('segmentations', self.dir, 'eye_mask_' + self.im_name + self.mask_ext)

        self.no_fin_segmentation=cv.imread(nf_seg_path,cv.IMREAD_GRAYSCALE)*255   
        self.full_segmentation=cv.imread(full_seg_path)*255   
        self.dorsal_mask=cv.imread(dorsal_seg_path,cv.IMREAD_GRAYSCALE)*255
        self.adipose_mask=cv.imread(adipose_seg_path,cv.  IMREAD_GRAYSCALE)*255  
        self.caudal_mask=cv.imread(caudal_seg_path,cv.IMREAD_GRAYSCALE)*255  
        self.anal_mask=cv.imread(anal_seg_path,cv.IMREAD_GRAYSCALE)*255  
        self.pelvic_mask=cv.imread(pelvic_seg_path,cv.IMREAD_GRAYSCALE)*255
        self.pectoral_mask=cv.imread(pectoral_seg_path,cv.IMREAD_GRAYSCALE)*255
        self.eye_mask=cv.imread(eye_seg_path,cv.  IMREAD_GRAYSCALE)*255
        
    def run(self):
        self.check_freezer()
            
        if self.frozen:
            self.re_run()
        
        else:
            self.get_measurements()
            self.get_scale(ds=1)
            self.segment_fish()
        
            if self.degenerate:
                print('degenerate!\n')
                self.no_fin_area=None
                self.FL = None
                self.area = None
                self.sector_areas=[None]*(2*self.n_steps)
                self.line_lengths=[None]*(2*self.n_steps)
            else: 
                self.level_fish()
                self.get_fin_clips()
                self.get_full_segmentations()
                self.get_no_fin_area()
                self.filet_fish(n_steps=self.n_steps,ord=self.ord)
                self.get_eye_diameter()
            
    def re_run(self):
        
        self.get_measurements()
        self.get_scale(ds=1)
        self.segment_fish()
        self.level_fish()
        print('level fish done\n')
        # self.get_fin_clips()
        print('\nthawing...\n')
        self.thaw()
        print('\nthawed\n')
        self.get_no_fin_area()
        print(self.no_fin_area)
        print('get no fin area done\n')
        self.filet_fish(n_steps=self.n_steps,ord=self.ord)
        print('filet done\n')
        self.get_eye_diameter()
        print('eye diameter done\n')
        
def main():
    from segment_anything import SamPredictor, sam_model_registry
    # example:
    image_path = os.path.join('sushi','example_fish', '110524FishID6c.jpg')


    sam = sam_model_registry['vit_l'](checkpoint="./sam_vit_l_0b3195.pth")
    predictor = SamPredictor(sam)   
    
    myFish = fish(image_path, predictor,write_masks=True)
    myFish.run()
    myFish.write_full_masks()
    myFish.write_fin_masks()
    
    # nfs = myFish.no_fin_segmentation
    # cnfs = convex_hull_image(nfs)
    # plt.imshow(cnfs)
    # plt.show()

    print('\n\ntotal non-fin sector area = ', np.sum(myFish.sector_areas,axis=0))
    print('total area = ', myFish.area)
    print('total non-fin area = ', myFish.no_fin_area)
    print('\npredicted fork length = ', myFish.FL)
    print('predicted eye diameter = ', myFish.eye_diameter, '\n\n')
    box = myFish.prediction_box
    slice = np.s_[box[1]:box[3], box[0]:box[2]]


    # Create a figure and a set of subplots
    save_figures = True # save figures
    
    fig, axes = plt.subplots(2, 2)

    # Display the first image on the left subplot
    axes[0,0].imshow(myFish.image[slice])
    axes[0,0].set_title('Raw Input')

    # Display the second image on the middle subplot
    axes[0,1].imshow(myFish.full_segmentation[slice]*255)
    axes[0,1].set_title('Segmentation')
    
    # Display the third image on the right subplot
    axes[1,1].imshow(np.sum(myFish.filled_sectors,axis=0)[slice])
    axes[1,1].set_title('Fileted Segmentation')
    # Display the third image on the right subplot
    axes[1,0].imshow(myFish.no_fin_segmentation[slice])
    axes[1,0].set_title('No Fin Segmentation')

    print(myFish.line_lengths)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    if save_figures:
        plt.savefig('fin_clipping_example.eps',dpi=200)
        plt.savefig('fin_clipping_example.png',dpi=200)

    # Show the plot
    plt.show()
    
    def create_figure(myFish):
        box = myFish.prediction_box
        slice = np.s_[box[1]:box[3], box[0]:box[2]]
    pass
    
if __name__ == '__main__':
    main()
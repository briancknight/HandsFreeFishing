import pandas as pd
import numpy as np
import cv2
import os
from segment_anything import SamPredictor, sam_model_registry
from HandsFreeFishing import preprocess_adult_steelhead, compute_contour, fin_clipping 
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
# from skimage.morphology import skeletonize, medial_axis

########### FIND THE 17TH LANDMARK ??? Should be linked to landmark 10... ###########
ord=30

def get_head_landmarks(fish, ord=10):
    head_contour, head_box_bounds =compute_contour(fish.head_mask,ord=ord)
    width=head_box_bounds[2]-head_box_bounds[0]
    # print('width=',width)
    signal = -head_contour[:,0]
    snout_peaks, properties = find_peaks(signal)
    peak_heights=signal[snout_peaks]
    sorted_indices_by_height = np.argsort(peak_heights)[::-1]
    snout_peaks = snout_peaks[sorted_indices_by_height]
    
    # if two maxima occur
    if len(snout_peaks) > 1:
        snout1 = head_contour[snout_peaks[0]]
        snout2 = head_contour[snout_peaks[1]]
        # print(snout1[0]/snout2[0])
        # print('snout-extrema check: ', np.abs(snout2[0]-snout1[0])/width)
        
        # if two maxima occur with mouth closed, choose leftmost point
        if np.abs(snout2[0]-snout1[0])/width > 0.4: 
            if snout1[0] < snout2[0]:
                snout_pt = snout1
            else:
                snout_pt = snout2 
        # if two maxima occur with mouth open, choose point with smaller y-value
        elif snout1[1] < snout2[1]:
            snout_pt = snout1
        else:
            snout_pt = snout2
    # one maxima occurs in x, this should be the point we care about
    else:
        snout_pt = head_contour[snout_peaks[0]]

    right_side_head_indices = np.argwhere(head_contour[:,0] > width/2).flatten()
    # print(right_side_head_indices)
    right_side_head_contour = head_contour[right_side_head_indices]
    top_of_head_pt = right_side_head_contour[np.argmin(right_side_head_contour[:,1])]
    bottom_of_head_pt = right_side_head_contour[np.argmax(right_side_head_contour[:,1])]
    # print('top of head point = ', top_of_head_pt)
    # print('bottom of head point = ', bottom_of_head_pt)
    
    return snout_pt, top_of_head_pt, bottom_of_head_pt # LM 1, 2, 17
    # landmark_points.append(snout_pt) # 15
    # landmark_points.append(top_of_head_pt) # 16
    # landmark_points.append(bottom_of_head_pt) # 17
  
def get_dorsal_landmarks(fish, ord=10):

    dorsal_fin_contour, dorsal_fin_box_bounds =compute_contour(fish.dorsal_mask,ord=ord)
    x_min = np.argmin(dorsal_fin_contour[:,0])
    x_max = np.argmax(dorsal_fin_contour[:,0])
    left_dorsal_pt = dorsal_fin_contour[x_min]
    right_dorsal_pt = dorsal_fin_contour[x_max]

    left_dorsal_no_fin_idx=np.argmin(np.linalg.norm(fish.nf_recon-left_dorsal_pt,axis=1))
    right_dorsal_no_fin_idx = np.argmin(np.linalg.norm(fish.nf_recon-right_dorsal_pt,axis=1))

    signal=-np.abs(left_dorsal_pt[0] - fish.nf_recon[:,0])
    x_peaks,_ = find_peaks(signal)
    
    # unlikely case where extrema appears at and end point *eyeroll*
    if len(x_peaks)==1:
        N=len(signal)
        idx_list=range(N)
        idx_list = np.roll(idx_list, 10)
        signal_rolled=np.roll(signal, 10)
        x_peaks_tmp,_ = find_peaks(signal_rolled)
        x_peaks = idx_list[x_peaks_tmp]

    peak_heights=signal[x_peaks]
    sorted_indices_by_height = np.argsort(peak_heights)[::-1]
    x_peaks = x_peaks[sorted_indices_by_height]
    peak1=x_peaks[0]
    peak2=x_peaks[1]

    if fish.nf_recon[peak1][1] < fish.nf_recon[peak2][1]:
        below_left_dorsal_pt = fish.nf_recon[peak2]
    else:
        below_left_dorsal_pt = fish.nf_recon[peak1]
    
    return left_dorsal_pt, right_dorsal_pt, below_left_dorsal_pt, (left_dorsal_pt + below_left_dorsal_pt)/2 # LM 3, 4 ???, ???
    # landmark_points.append(left_dorsal_pt) # LM 3
    # landmark_points.append(right_dorsal_pt) # LM 4
    
    # # landmark_points.append(fish.nf_recon[left_dorsal_no_fin_idx])
    # # landmark_points.append(fish.nf_recon[right_dorsal_no_fin_idx])
    
    # landmark_points.append(below_left_dorsal_pt) # LM 5
    # landmark_points.append((left_dorsal_pt + below_left_dorsal_pt)/2) # LM 6
    
    # return landmark_points
  
def get_adipose_landmarks(fish, ord=10):
    
    fin_contour, fin_box_bounds =compute_contour(fish.adipose_mask,ord=ord)

    leftmost_point_idx = np.argmin(fin_contour[:,0])
        
    left_adipose_pt = fin_contour[leftmost_point_idx]

    # landmark_points.append(left_adipose_pt) # LM 12
    
    return left_adipose_pt # LM 5

def get_caudal_landmarks(fish, ord=10):
    caudal_fin_contour, caudal_fin_box_bounds =compute_contour(fish.caudal_mask,ord=ord)
    signal=-caudal_fin_contour[:,0]
    x_peaks, properties = find_peaks(signal)
    peak_heights=signal[x_peaks]
    sorted_indices_by_height = np.argsort(peak_heights)[::-1]
    x_peaks = x_peaks[sorted_indices_by_height]
    caudal_pt1 = caudal_fin_contour[x_peaks[0]]
    caudal_pt2 = caudal_fin_contour[x_peaks[1]]

    if caudal_pt1[1] > caudal_pt2[1]:
        top_caudal_pt=caudal_pt2
        bottom_caudal_pt=caudal_pt1
    else:
        top_caudal_pt=caudal_pt1
        bottom_caudal_pt=caudal_pt2
        
    # now find a point in the middle:
    middle_y_value=(top_caudal_pt[1]+bottom_caudal_pt[1])/2

    if x_peaks[0] < x_peaks[1]:
        slc=np.s_[x_peaks[0]:x_peaks[1]]
    else:
        slc=np.s_[x_peaks[1]:x_peaks[0]]
        
    caudal_contour_segment=caudal_fin_contour[slc]
    n_ccs = len(caudal_contour_segment)

    middle_caudal_pt = caudal_contour_segment[int(round(n_ccs/2))]

    return top_caudal_pt, middle_caudal_pt, bottom_caudal_pt # LM 6, 7, 8
    # landmark_points.append(top_caudal_pt) # LM 7
    # landmark_points.append(bottom_caudal_pt) # LM 8
    # landmark_points.append(middle_caudal_pt) # LM 9
    
    # return landmark_points

def get_anal_landmarks(fish, ord=10):
    anal_fin_contour, anal_fin_box_bounds =compute_contour(fish.anal_mask,ord=ord)
    x_min = np.argmin(anal_fin_contour[:,0])
    y_min = np.argmin(anal_fin_contour[:,1])
    left_anal_pt = anal_fin_contour[x_min]
    right_anal_pt = anal_fin_contour[y_min]

    # left_anal_no_fin_idx=np.argmin(np.linalg.norm(fish.nf_recon-left_anal_pt,axis=1))
    # right_anal_no_fin_idx = np.argmin(np.linalg.norm(fish.nf_recon-right_anal_pt,axis=1))
    # landmark_points.append(left_anal_pt) # LM 1
    # landmark_points.append(right_anal_pt) # LM 2
    
    return left_anal_pt, right_anal_pt # LM ???, ???

def get_pelvic_landmarks(fish, ord=10):
    
    fin_contour, fin_box_bounds =compute_contour(fish.pelvic_mask,ord=ord)

    leftmost_point_idx = np.argmin(fin_contour[:,0])
        
    left_pelvic_pt = fin_contour[leftmost_point_idx]

    # landmark_points.append(left_pelvic_pt) # LM 11
    
    return left_pelvic_pt # LM 13

def get_pectoral_landmarks(fish, ord=10):
    
    fin_contour, fin_box_bounds =compute_contour(fish.pectoral_mask,ord=ord)

    leftmost_point_idx = np.argmin(fin_contour[:,0])
        
    left_pectoral_pt = fin_contour[leftmost_point_idx]
    
    return left_pectoral_pt # LM ???

def get_eye_landmarks(fish, ord=10):
    ####### make sure the left and right points are consistently ordered #######
    eye_contour, eye_box_bounds =compute_contour(fish.eye_mask,ord=ord)

    average_y = np.mean(eye_contour[:,1])
    signal=-np.abs(eye_contour[:,1] - average_y)
    eye_peaks,_ = find_peaks(signal)
    y_peaks, properties = find_peaks(signal)
    peak_heights=signal[y_peaks]
    sorted_indices_by_height = np.argsort(peak_heights)[::-1]
    y_peaks = y_peaks[sorted_indices_by_height]

    eye_pt1 = eye_contour[y_peaks[0]]
    eye_pt2 = eye_contour[y_peaks[1]]
    
    if eye_pt1[0] < eye_pt2[0]: # make sure the left and right side are ordered consistently
        return eye_pt1, eye_pt2 # LM 13, LM 14
        # landmark_points.append(eye_pt1) # LM 13
        # landmark_points.append(eye_pt2) # LM 14
    else:
        return eye_pt2, eye_pt1 # LM 13, LM 14
        # landmark_points.append(eye_pt2) # LM 13
        # landmark_points.append(eye_pt1) # LM 14
    
def get_landmark_points(fish,ord=ord):
    image = fish.image.copy()
    
    nf_recon, nf_box_bounds = compute_contour(fish.no_fin_segmentation, ord=100)
    lm1, lm2, lm14 = get_head_landmarks(fish, ord=ord)
    lm3, lm4, lm12, lm17 = get_dorsal_landmarks(fish, ord=ord)
    lm5 = get_adipose_landmarks(fish, ord=ord)
    lm6,lm7,lm8 = get_caudal_landmarks(fish, ord=ord)
    lm10, lm9 = get_anal_landmarks(fish, ord=ord)
    lm11 = get_pelvic_landmarks(fish, ord=ord)
    lm13 = get_pectoral_landmarks(fish, ord=ord)
    lm15, lm16 = get_eye_landmarks(fish, ord=ord)
    
    landmark_points=[lm1,lm2,lm3,lm4,lm5,lm6,lm7,
                     lm8,lm9,lm10,lm11,lm12,lm13,
                     lm14,lm15,lm16,lm17]
    
    return landmark_points

def save_landmark_image(fish, landmark_points, name='my_fish'):
    image=fish.image.copy()
    for (idx,point) in enumerate(landmark_points):
        (x,y) = tuple(point.astype(int))
        radius = 13
        color = (0, 255, 0) # Red color in BGR (Blue, Green, Red)
        thickness = -1 # -1 fills the circle

        # Use cv2.circle() to draw each point
        cv2.circle(image, (x,y), radius, color, thickness)
        cv2.putText(image, str(idx+1), (x+10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    
    print('saving...', name+'_land_mark_points.png')
    cv2.imwrite(name+'_land_mark_points.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
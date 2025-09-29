import os
import numpy as np
import cv2 as cv

# helper functions
def stack_mask(mask,color=None):
    if color is None:
        return np.stack([mask,mask,mask],axis=2)
    else:
        nil = np.zeros_like(mask)
        
        if isinstance(color,tuple):
            return np.stack([(mask>0) * color[2]/255, (mask>0) * color[1]/255, (mask>0)*color[1]/255],axis=2)
        if color=='r':
            return np.stack([mask,nil,nil],axis=2)
        elif color=='g':
            return np.stack([nil,mask,nil],axis=2)
        elif color=='b':
            return np.stack([nil,nil,mask],axis=2)

def bg_black_to_white(img):
    mask=np.linalg.norm(img,axis=-1)>0
    bg=1-mask
    white_bg = stack_mask(bg)
    
    return img + 255*white_bg

def get_FL_image(myFish,contour=False, contour_color=(107,126,221),bg='black'):
    img=np.copy(myFish.image)
    if contour:
        img = 255*get_cropped_annotated_img(myFish,contour_color=contour_color,bg=bg)
        
    FL_pt1=myFish.FL_points[0]
    FL_pt2=myFish.FL_points[1]
    if myFish.horiz_flip=='1':
        FL_pt1 = np.array([img.shape[1] - FL_pt1[0], FL_pt1[1]])
        FL_pt2 = np.array([img.shape[1] - FL_pt2[0], FL_pt2[1]])

    FL_pt1=FL_pt1.astype(int)
    FL_pt2=FL_pt2.astype(int)
    line_image = cv.line(img, FL_pt1, FL_pt2, 255, 15)

    return line_image

def get_cropped_annotated_img(myFish,contour_color=(107,126,221),bg='black',with_fins=True):
    
    img=np.copy(myFish.image)
    
    if with_fins:
        contour_img=get_fish_contour_img(myFish, myFish.recon)
        segmentation=stack_mask(myFish.fish_mask_full)/255
    else:
        contour_img=get_fish_contour_img(myFish, myFish.nf_recon)
        if myFish.no_fin_convex_hull_mask is None:
            segmentation=stack_mask(myFish.no_fin_segmentation)/255
        else:
            segmentation=stack_mask(myFish.no_fin_convex_hull_mask)/255
            
    contour_mask=stack_mask(contour_img)
    contour_mask_green=stack_mask(contour_img,color=contour_color)
    cropped_no_countour=(255-(contour_mask))/255
    segmented = img * segmentation
    # plt.imshow(segmented)
    # cropped_annotated=(cropped_no_countour*segmented) + 255*contour_mask_green

    cropped_annotated = (cropped_no_countour*segmented) + 255*contour_mask_green

    if bg == 'white':
        white_bg=(1-(np.linalg.norm(cropped_annotated,axis=-1)>0))
        white_bg = stack_mask(white_bg,(255,255,255))
        
        cropped_annotated = white_bg + cropped_annotated
        
    return cropped_annotated
    
def get_fish_contour_img(fish, contour,w=4):
    contour_idxs = (contour).astype(int)
    contour_img=np.zeros_like(fish.no_fin_segmentation)
    for idx in contour_idxs:
        contour_img[idx[1]-w:idx[1]+w,idx[0]-w:idx[0]+w] = 255
    if fish.horiz_flip=='1':
        contour_img=contour_img[:,::-1]
    return contour_img

# figure scripts

def save_truss_figure(myfish,im_name, figure_path='figures'):
    
    os.makedirs(figure_path, exist_ok=True)
    
    box = myfish.prediction_box
    slice = np.s_[box[1]:box[3], box[0]:box[2]]
    
    myfish.get_digitized_landmarks(n_steps=7)

    image = np.zeros_like(myfish.image)
    
    if myfish.horiz_flip=='1':    
        image = image[:,::-1]

    line_color=(0,0,255)
    line_thickness=3
    
    for start_end_pts in myfish.truss_start_end_pts:
        p1 = start_end_pts[0]
        p2 = start_end_pts[1]

        image = cv.line(image.astype(np.uint8), p1, p2, line_color, line_thickness)
    
    w=5
    for point in myfish.truss_points:
        (x,y) = point
        image[y-w:y+w,x-w:x+w] = np.array([255,0,255])
        
    if myfish.horiz_flip=='1':    
        image = image[:,::-1]
    
    image_mask = np.linalg.norm(image[slice],axis=-1) == 0
    stacked_image_mask = np.stack([image_mask, image_mask, image_mask], axis=-1)
    cv.imwrite(os.path.join(figure_path, im_name + '_truss.png'), cv.cvtColor(image[slice]+(myfish.image[slice] * stacked_image_mask),cv.COLOR_RGB2BGR))
        
def save_example_figures(myfish, im_name, figure_path = 'figures'):

        os.makedirs(figure_path, exist_ok=True)
        
        box = myfish.prediction_box
        slice = np.s_[box[1]:box[3], box[0]:box[2]]
        # cv.imwrite(im_names[i] + '_slice.png', cv.cvtColor(myfish.image[slice], cv.COLOR_BGR2RGB))

        if myfish.no_fin_convex_hull_mask is not None:
            cropped_mask = myfish.no_fin_convex_hull_mask * 255
        else:
            cropped_mask = myfish.no_fin_segmentation * 255
            
        cropped_mask = np.stack([cropped_mask, cropped_mask, cropped_mask], axis=-1)

        cropped_masked_black = myfish.image[slice]*(255*cropped_mask[slice])
        cropped_masked = bg_black_to_white(cropped_masked_black)

        # original crop
        cv.imwrite(os.path.join(figure_path, im_name + '_slice.png'), cv.cvtColor(bg_black_to_white(myfish.image[slice]).astype(np.uint8), cv.COLOR_BGR2RGB))

        contour_color = 'g'
        bg='white'
        
        # full segmentation
        cropped_contour_img = get_cropped_annotated_img(myfish, contour_color=contour_color,bg=bg, with_fins=True)
        cv.imwrite(os.path.join(figure_path, im_name + '_full_seg_contour.png'), cv.cvtColor((255*cropped_contour_img[slice]).astype(np.uint8), cv.COLOR_BGR2RGB))
        
        cv.imwrite(os.path.join(figure_path, im_name + '_full_seg.png'), bg_black_to_white(255*myfish.full_segmentation[slice]).astype(np.uint8))
        
        # no fin segmentation 
        nf_cropped_contour_img = get_cropped_annotated_img(myfish, contour_color=contour_color,bg=bg,with_fins=False)
        cv.imwrite(os.path.join(figure_path, im_name + '_nf_cropped_contour.png'), cv.cvtColor((255*nf_cropped_contour_img[slice]).astype(np.uint8), cv.COLOR_BGR2RGB))
        
        # fork length visual
        FL_image = get_FL_image(myfish,contour=True,contour_color='g')
        cv.imwrite(os.path.join(figure_path, im_name + '_FL_image.png'), cv.cvtColor(bg_black_to_white(FL_image[slice]).astype(np.uint8), cv.COLOR_BGR2RGB))

        # eye diameter visual
        cropped_mask_initial = 255 * (myfish.fish_mask > 0)
        cropped_mask_initial_stacked = np.stack([cropped_mask_initial, cropped_mask_initial, cropped_mask_initial], axis=-1)
        cropped_masked_initial_black = myfish.image[slice] * (cropped_mask_initial_stacked[slice])
        
        eye_contour_img = get_fish_contour_img(myfish, myfish.eye_contour)
        contour_img=get_fish_contour_img(myfish, myfish.recon)

        eye_contour_mask=stack_mask(eye_contour_img,color='r')
        contour_mask=stack_mask(contour_img,'g')

        all_contours=stack_mask(eye_contour_img+contour_img)
        cropped_no_countour=(255-(all_contours))
        cropped_annotated=cropped_no_countour[slice]*cropped_masked_initial_black + contour_mask[slice] + eye_contour_mask[slice]
        
        cv.imwrite(os.path.join(figure_path, im_name + '_eye_image.png'), cv.cvtColor(bg_black_to_white(cropped_annotated).astype(np.uint8), cv.COLOR_BGR2RGB))
        
        # partitioned surface area visual
        partitioned_SA_mask = np.sum([(myfish.filled_partitions[i][slice] > 0) * (0.2*(i+1)) for i in range(len(myfish.filled_partitions))],axis=0)
        partitioned_SA_mask = np.stack([partitioned_SA_mask, np.zeros_like(partitioned_SA_mask), np.zeros_like(partitioned_SA_mask)], axis=-1) * 255
        cv.imwrite(os.path.join(figure_path, im_name + '_partitioned_SA_image.png'), cv.cvtColor(bg_black_to_white(partitioned_SA_mask).astype(np.uint8), cv.COLOR_BGR2RGB))
         
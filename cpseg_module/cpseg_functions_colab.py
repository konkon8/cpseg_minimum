import numpy as np
import pandas as pd

from PIL.Image import fromarray
from skimage.measure import label, regionprops_table
from skimage.morphology import disk, remove_small_objects, skeletonize
from skimage.segmentation import clear_border
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import ruptures as rpt
import cv2
from cv2 import morphologyEx, MORPH_CLOSE
from joblib import Parallel, delayed
from glob import glob
import time
from random import randint
from skan import Skeleton, summarize, skeleton_to_csgraph
from scipy import interpolate

def rot_edge(irot,values):
    # Variables
    rot_dgree_list, I, edge_mean, numBorder, model, y_interval, penalty_value, changepoint_algo = values
    I_Height, I_Width = I.shape  # image shape

    # Image rotation
    rot_degree = rot_dgree_list[irot]  # Rotation angle(degree) clockwise:minus
    im = fromarray(I)  # Convert to pillow object
    J_plt = im.rotate(rot_degree, expand=True, fillcolor=edge_mean)  # Rotate
    J_np = np.array(J_plt)  # Convert to array
    J = J_np

    # Border detection
    BW1 = ak_get_edge(J, numBorder, model, y_interval, penalty_value, changepoint_algo)

    # Inverse rotation of the border
    BW1_imobject = fromarray(BW1)

    inv_rot_degree = np.copysign(rot_degree, -1).astype(np.int64)
    BW1_imobject_rot = BW1_imobject.rotate(inv_rot_degree, expand=True)  # inverse rotate
    BW = np.array(BW1_imobject_rot).astype(int)  # Convert to int nparray

    # Crop a rectangle at the image center to the size of the original image
    if rot_degree == 0 or rot_degree == 180:  # If the size of the rotated image is same as the original
        BW12 = BW1
    else:
        left = int((BW.shape[1] - I_Width) / 2)  # J_Width # Positions of the corners
        top = int((BW.shape[0] - I_Height) / 2)  # J_Height
        right = left + I_Width
        bottom = top + I_Height
        BW12 = BW[top:bottom, left:right]  # Crop a rectangle at the image center

    return BW12

def ak_rotate_edge_overlay(I,numLayer,rotation_angle,numBorder,model,y_interval,penalty_value, changepoint_algo):
    """
    Created on Fri Apr 16 07:26:17 2021

    'ak_rotate_edge_overlay' rotate an image, detect edges, and inversely
    rotate the edge. If multiple rotation angles were specified, each edges
    were added or stacked along a new axiｓ.'I' must be an 2-dimensional array
    which represent gray scale image.
    """
    #%%

    # parameters
    # numLayer       = 0  # 0: single
    #                     # 1: multiple
    # rotation_angle = 1  # 0: 45 degree, 3: semicircle
    #                     # 1: 30 degree, 4: semicircle
    #                     # 2: 15 degree, 5: semicircle
    # numBorder      = 2  # Number of change points
    # dY             = 3  # Hight of each rectangle
    # region         = 0  # 0: calculate all y, 1: calculate 1/yRatio along y
    # yRatio         = 5  # if region = 1,calculate 1/yRatio along y

    # Processing
    # Rotation angle (degree)

    if rotation_angle == 0:
        rot_degree_list = np.array([0,45,90,135,180,225,270,315]) # Every 45˚
    elif rotation_angle == 1:
        rot_degree_list = np.array([0,30,60,90,120,150,180,210,240,270,300,330]) # Every 30˚
    elif rotation_angle == 2:
        rot_degree_list = np.array([0,15,30,45,60,75,90,105,120,135,150,165,180,195,\
                           210,225,240,255,270,285,300,315,330,345]) # Every 15˚
    elif rotation_angle == 3:
        rot_degree_list = np.array([0,45,90,135]) # Every 45˚, semicircle
    elif rotation_angle == 4:
        rot_degree_list = np.array([0,30,60,90,120,150]) # Every 30˚, semicircle
    elif rotation_angle == 5:
        rot_degree_list = np.array([0,15,30,45,60,75,90,105,120,135,150,165]) # Every 15˚, semicircle

    # Mean pixel value at the edge of the image
    img_ori = I
    I = np.array(I)
    edge = np.hstack((I[9,:], I[-10,:], I[:,9].T, I[:,-10].T))
    edge_mean = int(np.mean(edge).item())

    BW_final = np.zeros(I.shape)          # empty border object

    values = rot_degree_list, I, edge_mean, numBorder, model, y_interval, penalty_value, changepoint_algo

    result = Parallel(n_jobs=-1)([delayed(rot_edge)(i_rot, values) for i_rot in np.arange(len(rot_degree_list))])
    # Overlay images
    bw_each = np.array(result)
    bw_overlay = np.sum(bw_each, axis=0)
    if numLayer == 0:
        BW_final = bw_each
    elif numLayer == 1:
        BW_final = bw_overlay
    else:
        print('The variable "numLayer" should be either 0 or 1.')

    # Normalize(binalize)
    BW_final = np.where(BW_final != 0, 1, 0)  # Replace non-zero elements to 1

    return BW_final

def ak_get_edge_xpostion(iy,values):
    # Crop image as rectangle w/ hight of one pixel
    I, numBorder, model, xBorder, penalty_value, changepoint_algo = values
    intensity_iy = I[iy, :]  # iy: y position

    # Detect change points(x)
    algo = rpt.KernelCPD(kernel=model, min_size=2, jump=1).fit(intensity_iy) # default: min_size=2
    if changepoint_algo == 1: # Number of border position not known
        my_bkps = algo.predict(pen=penalty_value)
        #
        repeat_bkps = 1
        my_bkps = algo.predict(pen=penalty_value)
        while repeat_bkps in np.arange(30) and len(my_bkps) - 1 > my_bkps[-1] /10:
            my_bkps = algo.predict(pen=penalty_value)
            repeat_bkps += 1

        xBorder = np.floor(np.array(my_bkps[0:-1])) # Avoid total indexes at the end

        if xBorder.size == 0:
            xpos = 0
        else:
            xpos = xBorder[0:xBorder.size].T
    elif changepoint_algo == 0: # Number of border position fixed
        my_bkps = algo.predict(n_bkps=numBorder)
        xBorder = np.floor(np.array(my_bkps[0:-1])) # Avoid total indexes at the end
        xpos = xBorder[0:numBorder].T

    return xpos

def ak_get_edge(I, numBorder, model, y_interval, penalty_value, changepoint_algo):
    """
    Created on Fri Apr 16 07:08:16 2021

    'ak_get_edge' detects edges in an image.
    """

    t_e0 = time.time()
    # Arguments
    Iori = np.array(I)  # Convert to image to numpy array
    I = Iori

    # Initialization of variables
    if changepoint_algo == 0: # border number fixed
        numBorder = numBorder
    elif changepoint_algo == 1: # border number not known
        numBorder = I.shape[1] # width of the original image
    xBorder = np.zeros(numBorder)  # edge position x coordinates
    xChangePoint = np.zeros((I.shape[0], numBorder))  # edge position coordinates [y,x]

    # Gaussian blur
    sigma = 3
    I_blur = cv2.GaussianBlur(I, (5,1), sigma)  # Gaussian blur
    I = I_blur

    # Processing
    values = [I, numBorder, model, xBorder, penalty_value, changepoint_algo]
    y_scan_index = np.arange(0, I.shape[0], y_interval)

    # Edge detection
    result = np.zeros((len(y_scan_index),numBorder)).astype(int)
    ind_list = np.arange(0,len(y_scan_index))
    for i in ind_list:
        xpos_i = np.array(ak_get_edge_xpostion(y_scan_index[i], values)).astype(int)
        result[i, 0:xpos_i.size] = xpos_i
    xChangePoint[y_scan_index, 0:numBorder] = result
    xChangePoint = xChangePoint.astype(int)

    # Get border coordinate
    subVector_h1 = np.tile(np.arange(I.shape[0]), numBorder)
    subVector_h2 = np.ravel(xChangePoint, order='F').astype(np.int64)
    BW = np.zeros((I.shape[0], I.shape[1]))
    my_index = np.arange(len(subVector_h1)).astype(np.int64)


    BW[subVector_h1[my_index], subVector_h2[my_index]] = 1

    # Remove edges at the border
    BW[:9,:]           = 0
    BW[-10:-1,:]       = 0
    BW[:,0:9]          = 0
    BW[:,-10:-1]       = 0

    t_e1 = time.time()
    elapsed_time = t_e1 - t_e0
    #print(f'Edge detection time(sec): {elapsed_time}')

    return BW

def ak_post_process(BW,smallNoiseRemov,senoise,neib,select_biggest,nskeletonize):
    """
    Post processing edge maps by removing small noises etc.
    
    Parameters
    ----------
    BW : numpyndarray, int
         2D Edge map
    smallNoiseRemov : int
         0:No noise removal
         1:Connect border, then remove small noises
         2:Remove small noisess first, then connect
    senoise : int
         Structural element size
    neib : int
         Area of imclose, pix
    select_biggest : int
         0: do not select
         1: select biggest
    nskeletonize : int
         0: No skeletonization
         1: Skeletonize
    
    Returns BWlast
    -------
    
    """

    # Remove signal at the edge
    #BW = clear_border(BW) # remove artifacts connected to image border
   
    # Remove small objects
    BW = BW.astype('uint8')
    se_noiseremove = disk(senoise) #structual elements, circle    
    if smallNoiseRemov == 0: # No noise removal
        BWs = BW
    elif smallNoiseRemov == 1:  #Connect border -> Remove small noise      
        BW1 = morphologyEx(BW, MORPH_CLOSE, se_noiseremove) # Connect border
        BW1_labels = label(BW1)
        BWs = remove_small_objects(BW1_labels, min_size=neib) # Remove small object
       
    elif smallNoiseRemov == 2:  # Remove small noises -> Connect border
        BW_labels = label(BW)
        BW1 = remove_small_objects(BW_labels, min_size=neib) # Remove small object
        BWs = morphologyEx(BW1, MORPH_CLOSE, se_noiseremove) # Connect border
    else:
        print('smallNoiseRemov should be 0,1,2.')
    BWs = np.where(BWs != 0, 1, 0)
   
    # Select largest component (segmentation)
    if select_biggest == 0:
        BWs_biggest = BWs
    elif select_biggest == 1:
        label_BWs = label(BWs, return_num=True)  # Label connected components
        label_image = label_BWs[0]  # labeled image
        properties = ['label', 'area']
        df = pd.DataFrame(regionprops_table \
                              (label_image, properties=properties))  # Data frome of area and label
        df_area_max = np.max(df.area)  # Area of largest component
        max_index = np.array(np.where(df.area == df_area_max))  # label of the largest component
        label_image_largest = np.where(label_image == (max_index + 1), 1,
                                       0)  # replace the largest region w/ 1 and others with 0
        BWs_biggest = np.where(label_image_largest != 0, 1, 0)  # Replace non-zero elements to 1
    else:
        print('BWs_biggest should be 0 or 1.')

    # Reduce object to 1-pixel wide curved lines
    if nskeletonize == 0:
        BW_skel = BWs_biggest
    elif nskeletonize == 1:
        BW_skel = skeletonize(BWs_biggest)
    else:
        print('nskeletonize should be 0 or 1.')    
    
    BWlast = BW_skel
    
    return BWlast

def postprocess(bw_img, min_j2e_size=1000000):
    # Dilate edge
    dilation_repeats = 4
    kernel_size = 3
    disk_dil = disk(kernel_size)

    bw_img_uint8 = bw_img.astype('uint8')
    bw_dil = cv2.dilate(bw_img_uint8, disk_dil, iterations=dilation_repeats)
    skeleton = skeletonize(bw_dil).astype(np.uint8)

    # fig, ax = plt.subplots()  # test
    # ax.imshow(skeleton, cmap='gray')  # test
    # ax.set_title(f'Post-processing input')  # test
    # plt.show()# test

    # Select largest connected component except background
    _, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton)
    area = stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    #print(f'Number of connected components: {area.size}')#test
    if area.size > 1:
        skeleton2 = np.where(labels == area.argsort()[-2], 1, 0)  # Avoid the largest component which is background

        # fig, ax = plt.subplots()  # test
        # ax.imshow(skeleton2, cmap='gray')  # test
        # ax.set_title(f'Select 2nd largest connected component')  # test
        # plt.show()  # test

        # # Prune branches
        # _, _, stats2, _ = cv2.connectedComponentsWithStats(skeleton2.astype('int8'))
        # if np.count_nonzero(stats2[:,4] > 10) > 1: # If the number of component with height > 10 pixel is more than 2
        #     skeleton2 = prune_branch(skeleton2, min_j2e_size).astype(np.uint8)
        #
        #     # fig,ax = plt.subplots()#test
        #     # ax.imshow(skeleton2, cmap='gray')#test
        #     # ax.set_title(f'prune_branch_{rep_i}') #test
        #     # plt.show()  # test

    elif area.size <= 1: # Number of connected components
        skeleton2 = np.copy(skeleton)

    # Prune branches
    skeleton2 = prune_branch(skeleton2, min_j2e_size).astype(np.uint8)

    return skeleton2

def prune_branch(skeleton, min_j2e_size = 1000000):
    if np.count_nonzero(skeleton == 1) <= 10:
        skeleton1 = np.copy(skeleton)
    elif np.count_nonzero(skeleton == 1) > 10:
        # Get skeleton info
        skeleton_obj = Skeleton(skeleton)
        df = summarize(skeleton_obj)
        branch_list = df[df["branch-type"] == 1]  # List of junction-to-endpoint paths
        j2e_idx = df.index.values[(df['branch-type'] == 1) & (df['branch-distance'] < min_j2e_size)]  #junction-to-endpoint less than min_j2e_size(pixels)

        if skeleton_obj.n_paths == 1: # Only 1 path in the object
            skeleton1 = skeleton_obj.skeleton_image # No pruning
        elif skeleton_obj.n_paths >= 2:
            nPrune = 0
            #if np.any(j2e_idx) == True: #If j2e paths exist
            #while np.any(j2e_idx) == True:
            while np.size(j2e_idx) >= 1:
            #while branch_list.loc[:, 'branch-distance'].min() < min_j2e_size: # Filter with branch length, while min j2e path < min_j2e_size
                for i in np.arange(np.size(j2e_idx)):
                    raw_list = Skeleton.path_coordinates(skeleton_obj, j2e_idx[i]).astype('uint16')
                    for j in np.arange(raw_list.shape[0]):
                        skeleton[raw_list[j][0]][raw_list[j][1]] = 0 # Update skeleton

                # fig,ax = plt.subplots()#test
                # ax.imshow(skeleton, cmap='gray')#test
                # ax.set_title(f'pruned') #test
                # plt.show()  # test

                # Fill gaps
                skeleton = cv2.dilate(skeleton.astype(np.uint8), disk(1), iterations=1) # Dilation

                # fig,ax = plt.subplots()#test
                # ax.imshow(skeleton, cmap='gray')#test
                # ax.set_title(f'Dilated') #test
                # plt.show()  # test

                skeleton = skeletonize(skeleton).astype(np.uint8) # Skeletonize

                # fig,ax = plt.subplots()#test
                # ax.imshow(skeleton, cmap='gray')#test
                # ax.set_title(f'Skeletonized') #test
                # plt.show()  # test

                # Updated skeleton info
                if np.count_nonzero(skeleton == 1) <= 10:
                    j2e_idx = []
                    skeleton1 = np.zeros_like(skeleton)
                elif np.count_nonzero(skeleton == 1) > 10:
                    skeleton_obj = Skeleton(skeleton)
                    df = summarize(skeleton_obj)
                    branch_list = df[df["branch-type"] == 1] # List of junction-to-endpoint paths after the pruning
                    j2e_idx = df.index.values[(df['branch-type'] == 1) & (df['branch-distance'] < min_j2e_size)]
                    skeleton1 = np.copy(skeleton)
                nPrune += 1
                print(f'prune repeat: {nPrune}')

                # fig,ax = plt.subplots()#test
                # ax.imshow(skeleton, cmap='gray')#test
                # ax.set_title(f'Pruning repeat: {nPrune}')#test
                # plt.show()  # test

            else:
                skeleton1 = skeleton_obj.skeleton_image


    return skeleton1

def find_ends(skeleton):
    _, _, stats, _ = cv2.connectedComponentsWithStats(skeleton)
    if np.count_nonzero(skeleton == 1) > 10:
        skeleton0 = Skeleton(skeleton)
        endpoint_coordinates = skeleton0.coordinates[np.nonzero(skeleton0.degrees == 1)] # degree: number of neighbouring pixels
    elif np.count_nonzero(skeleton == 1) <= 10:
        endpoint_coordinates = np.zeros(1)

    return endpoint_coordinates

def connect_ends(skeleton1, min_j2e_size = 1000000):
    # Classify
    #skeleton1 = prune_branch(skeleton, min_j2e_size)

    # Connect_ends
    end_coordinates = find_ends(skeleton1)

    skeleton2 = np.copy(skeleton1)
    if np.shape(end_coordinates)[0] == 2:
        e1 = end_coordinates[0,:].astype(int)
        e2 = end_coordinates[1,:].astype(int)
        x = [e1[1], e2[1]]
        y = [e1[0], e2[0]]
        if abs(e1[1] - e2[1]) >= abs(e1[0] - e2[0]):
            point = np.abs(e1[1] - e2[1]).astype(int)
            f = interpolate.interp1d(x, y)
            X = np.linspace(e1[1], e2[1], num=point + 1, endpoint=True).astype(int)
            Y = f(X)
            Yr = np.round(Y).astype(int)
            index = np.stack([Yr, X], axis=1)
            index = index[~np.isnan(index.any(axis=1)),:]
            for i in np.arange(index.shape[0]):
                skeleton2[index[i,0],index[i,1]] = 1
        else:
            point = np.abs(e1[0] - e2[0]).astype(int)
            f = interpolate.interp1d(y, x)
            Y = np.linspace(e1[0], e2[0], num=point + 1, endpoint=True).astype(int)
            X = f(Y)
            Xr = np.round(X).astype(int)
            index = np.stack([Y, Xr], axis=1)
            index2 = index[~np.isnan(index.any(axis=1)),:]
            for i in np.arange(index2.shape[0]):
                skeleton2[index2[i, 0], index2[i, 1]] = 1

    return skeleton2
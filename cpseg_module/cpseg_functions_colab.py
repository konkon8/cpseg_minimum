import numpy as np
import pandas as pd
from PIL.Image import fromarray
from skimage.measure import label, regionprops_table
from skimage.morphology import disk, remove_small_objects, skeletonize
from skimage.segmentation import clear_border
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import cv2
from cv2 import morphologyEx, MORPH_CLOSE
from PIL.Image import fromarray #, BILINEAR, CUBIC
import ruptures as rpt
from os import sep
import os
from glob import glob
from random import randint
from joblib import Parallel, delayed
import time
from skan import Skeleton, summarize, skeleton_to_csgraph
from scipy import interpolate
import scipy
import itertools

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

def postprocess(bw_img, min_j2e_size=1000000, cycle_max_pixel = 500, remove_cycle_var = 1):
    # Dilate edge
    dilation_repeats = 4
    kernel_size = 3
    disk_dil = disk(kernel_size)

    bw_img_uint8 = bw_img.astype('uint8')
    bw_dil = cv2.dilate(bw_img_uint8, disk_dil, iterations=dilation_repeats)
    skeleton = skeletonize(bw_dil).astype(np.uint8)

    # Select largest connected component except background
    _, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton)
    area = stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    if area.size > 1:
        skeleton2 = np.where(labels == area.argsort()[-2], 1, 0)  # Avoid the largest component which is background
    elif area.size <= 1: # Number of connected components
        skeleton2 = np.copy(skeleton)

    # Remove small cycles
    if remove_cycle_var == 1:
        skeleton2_fs = remove_cycle(skeleton2, cycle_max_pixel)
    elif remove_cycle_var == 0:
        skeleton2_fs = np.copy(skeleton2)

    # Prune branches
    skeleton3 = prune_branch(skeleton2_fs, min_j2e_size).astype(np.uint8)

    return skeleton3

def prune_branch(skeleton, min_j2e_size = 1000000):
    if np.count_nonzero(skeleton == 1) <= 10: # Very small skeletons
        skeleton1 = np.copy(skeleton)
    elif np.count_nonzero(skeleton == 1) > 10:
        # Get skeleton info
        skeleton_obj = Skeleton(skeleton)
        df = summarize(skeleton_obj)
        j2e_idx = df.index.values[(df['branch-type'] == 1) & (df['branch-distance'] < min_j2e_size)]  #junction-to-endpoint less than min_j2e_size(pixels)
        if skeleton_obj.n_paths == 1: # Only 1 path in the object
            skeleton1 = skeleton_obj.skeleton_image # No pruning
        elif skeleton_obj.n_paths >= 2: # More than 2 paths in the object
            nPrune = 0
            while np.size(j2e_idx) >= 1:
                for i in np.arange(np.size(j2e_idx)):
                    raw_list = Skeleton.path_coordinates(skeleton_obj, j2e_idx[i]).astype('uint16') #Get path coordinates
                    for j in np.arange(raw_list.shape[0]):
                        skeleton[raw_list[j][0]][raw_list[j][1]] = 0 # Erase branches
                # Fill gaps
                skeleton = cv2.dilate(skeleton.astype(np.uint8), disk(1), iterations=1) # Dilation
                skeleton = skeletonize(skeleton).astype(np.uint8) # Skeletonize
                if np.count_nonzero(skeleton == 1) <= 10:
                    j2e_idx = []
                    skeleton1 = np.zeros_like(skeleton)
                elif np.count_nonzero(skeleton == 1) > 10:
                    skeleton_obj = Skeleton(skeleton)
                    df = summarize(skeleton_obj)
                    j2e_idx = df.index.values[(df['branch-type'] == 1) & (df['branch-distance'] < min_j2e_size)]
                    skeleton1 = np.copy(skeleton)
                nPrune += 1
                print(f'prune repeat: {nPrune}')

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

def connect_ends(edge,skeleton1, min_j2e_size = 1000000, connect_method = 2):
    end_coordinates = find_ends(skeleton1)
    skeleton2 = np.copy(skeleton1)

    if np.shape(end_coordinates)[0] == 0: # No ends > Do nothing
        skeleton2 = np.copy(skeleton1)
    elif np.shape(end_coordinates)[0] == 1: # One ends = one j2e > erase j2e
        if np.count_nonzero(skeleton1 == 1) > 10:
            skeleton1_obj = Skeleton(skeleton1) # Get skeleton object
            df1 = summarize(skeleton1_obj) # Get dataframe
            j2e1_idx = df1.index.values[(df1['branch-type'] == 1)]  # Get id of junction-to-endpoint path
            if np.size(j2e1_idx) == 1:
                j2e_coordinates = Skeleton.path_coordinates(skeleton1_obj, j2e1_idx[0]).astype('uint16') #Get path coordinates
                for i in np.arange(j2e_coordinates.shape[0]):
                    skeleton1[j2e_coordinates[i][0]][j2e_coordinates[i][1]] = 0  # Erase branches
            skeleton2 = np.copy(skeleton1)

    # %% Two ends
    elif np.shape(end_coordinates)[0] == 2: # Two ends
        skeleton1_obj = Skeleton(skeleton1) # Get skeleton object
        df1 = summarize(skeleton1_obj) # Get dataframe
        j2e1_idx = df1.index.values[(df1['branch-type'] == 1)]
        if np.size(j2e1_idx) == 0: # No j2e -> Connect ends, path -> 100 pixels
            print(f'pixel number is: {np.count_nonzero(skeleton1 == 1)}')
            if np.count_nonzero(skeleton1 == 1) >= 500:
                e1 = end_coordinates[0,:].astype(int)
                e2 = end_coordinates[1,:].astype(int)
                # test
                avr_edge_value, index = get_edge_density(edge, e1, e2)
                if avr_edge_value > 20:
                    # Fill gap with 1
                    for i in np.arange(index.shape[0]):
                        skeleton2[index[i, 0], index[i, 1]] = 1
                    print(f'Ends were connected')
            else:
                skeleton2 = np.zeros_like(skeleton1)
        elif np.size(j2e1_idx) >= 1: # Two ends with end-to-junctions > Erase branches
            for i in np.arange(np.size(j2e1_idx)):
                if df1['branch-distance'].values[j2e1_idx[i]] < 200:  # Max size of pruning branch
                    j2e_coordinates = Skeleton.path_coordinates(skeleton1_obj, j2e1_idx[i]).astype('uint16')  # Get path coordinates
                    for j in np.arange(j2e_coordinates.shape[0]):
                        skeleton1[j2e_coordinates[j][0]][j2e_coordinates[j][1]] = 0  # Erase branches
                    skeleton2 = np.copy(skeleton1)

                    # Remove resulting branch
                    # Dilate edge
                    dilation_repeats = 4
                    kernel_size = 3
                    disk_dil = disk(kernel_size)
                    skeleton2_uint8 = skeleton2.astype('uint8')
                    skeleton2 = cv2.dilate(skeleton2_uint8, disk_dil, iterations=dilation_repeats)
                    skeleton2 = skeletonize(skeleton2).astype(np.uint8)
                    skeleton2 = remove_j2e(skeleton2)

    #%% Three ends
    elif np.shape(end_coordinates)[0] >= 3: # Three ends
        if connect_method == 0: # Longest shortest path
            skeleton0 = get_lsp(skeleton1)
            # Connect ends
            if np.shape(find_ends(skeleton0))[0] == 2:  # Two ends
                skeleton0_obj = Skeleton(skeleton0)  # Get skeleton object
                df0 = summarize(skeleton0_obj)  # Get dataframe
                j2e0_idx = df0.index.values[(df0['branch-type'] == 1)]
                if np.size(j2e0_idx) == 0:  # No j2e > Connect ends, path > 100 pixels
                    print(f'pixel number is: {np.count_nonzero(skeleton1 == 1)}')
                    if np.count_nonzero(skeleton0 == 1) >= 500:
                        end_coordinates = find_ends(skeleton0)
                        e1 = end_coordinates[0, :].astype(int)
                        e2 = end_coordinates[1, :].astype(int)
                        # test
                        avr_edge_value, index = get_edge_density(edge, e1, e2)
                        if avr_edge_value > 20:
                            # Fill gap with 1
                            for i in np.arange(index.shape[0]):
                                skeleton0[index[i, 0], index[i, 1]] = 1
                            print(f'Ends were connected')
            skeleton2 = np.copy(skeleton0)
        elif connect_method == 1: # Edge density
            # Get ends
            ends_comb = list(itertools.combinations(np.arange(3), 2))
            # Get edge density
            avr_edge_value = np.zeros(3)
            for i, i_ends_comb in enumerate(ends_comb):
                e0_i = end_coordinates[i_ends_comb[0]].astype(int)
                e1_i = end_coordinates[i_ends_comb[1]].astype(int)
                avr_edge_value[i], _ = get_edge_density(edge, e0_i, e1_i)
            max_id = np.argmax(avr_edge_value)
            max_comb_id = ends_comb[max_id]
            _, index_connect = get_edge_density(edge,end_coordinates[max_comb_id[0],:],
                                     end_coordinates[max_comb_id[1], :])
            # Fill the gap with 1
            for i in np.arange(index_connect.shape[0]):
                skeleton2[index_connect[i, 0], index_connect[i, 1]] = 1
            print(f'Ends were connected')

        elif connect_method == 2: # Distance
            # Get ends
            ends_comb = list(itertools.combinations(np.arange(np.shape(end_coordinates)[0]),2))
            # Get end distances
            distances = np.zeros(len(ends_comb))
            for i, i_ends_comb in enumerate(ends_comb):
                distances[i] = np.linalg.norm(end_coordinates[i_ends_comb[0], :].astype(int) \
                                              - end_coordinates[i_ends_comb[1], :].astype(int)).astype(int)
                print(f'pair {i_ends_comb}, id {i}, distance {distances[i]}')
            min_id = np.argmin(distances)
            min_comb_id = ends_comb[min_id]
            _, index_connect = get_edge_density(edge,end_coordinates[min_comb_id[0],:],
                                     end_coordinates[min_comb_id[1], :])

            # Fill the gap with 1
            if distances[min_id] < 200:
                for i in np.arange(index_connect.shape[0]):
                    skeleton2[index_connect[i, 0], index_connect[i, 1]] = 1
                print(f'Ends were connected')

        # Remove resulting branch
        skeleton2 = remove_j2e(skeleton2)

    # Dilate edge
    dilation_repeats = 4
    kernel_size = 3
    disk_dil = disk(kernel_size)
    skeleton2_uint8 = skeleton2.astype('uint8')
    skeleton2 = cv2.dilate(skeleton2_uint8, disk_dil, iterations=dilation_repeats)
    skeleton2 = skeletonize(skeleton2).astype(np.uint8)
    skeleton2 = remove_j2e(skeleton2)

    return skeleton2

def remove_cycle(img_binary, cycle_max_pixel = 500):
    img_binary = img_binary.astype('uint8')
    img_origin = np.zeros_like(img_binary).astype('uint8')
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        id_list_innerlayer = np.nonzero(hierarchy[0, :, 3] != -1) # contours with parents
        # contours smaller than size_limit_pixel
        id_list_size = []
        for i in range(len(contours)):
            if np.size(contours[i]) < cycle_max_pixel:
                id_list_size.append(i)

        id_list = np.intersect1d(id_list_innerlayer, np.array(id_list_size)) # contour id with parents smaller than size_limit (pixels)
        cycles = np.zeros_like(img_binary).astype('uint8')
        for i in np.arange(np.size(id_list)):
            cycle_i = cv2.drawContours(img_origin,      # Plot empty image
                                     contours,          # contour info
                                     id_list[i],        # Contour id
                                     (255, 255, 255),   # Color
                                     1)                 # Contour width

            cycle0 =  (cycle_i / 255).astype('uint8')
            cycles = cycles | cycle0 # Add newly selected cycles to previous

        # Fill cycles
        cycle_fill = scipy.ndimage.binary_fill_holes(cycles).astype(int)
        img_fill = img_binary | cycle_fill # Add original edge with filled cycles
    else:
        img_fill = img_binary

    img_fill = img_fill.astype(np.uint8)
    cycle_removed = skeletonize(img_fill).astype(np.uint8)

    return cycle_removed

def get_edge_density(edge, e1, e2):
    # Dilate edge
    dilation_repeats = 4
    kernel_size = 3
    disk_dil = disk(kernel_size)
    edge_uint8 = edge.astype('uint8')
    edge_dil = cv2.dilate(edge_uint8, disk_dil, iterations=dilation_repeats)

    x = [e1[1], e2[1]]
    y = [e1[0], e2[0]]
    if abs(e1[1] - e2[1]) >= abs(e1[0] - e2[0]):
        point = np.abs(e1[1] - e2[1]).astype(int)
        f = interpolate.interp1d(x, y)
        X = np.linspace(e1[1], e2[1], num=point + 1, endpoint=True).astype(int)  # x cooridinate
        Y = f(X)
        Yr = np.round(Y).astype(int)  # y cooridinate
        index = np.stack([Yr, X], axis=1)
        index = index[~np.isnan(index.any(axis=1)), :]  # coordinates (x,y)
        # Get average edge number
        pixels_between_ends = index.shape[0]
        sum_edge_value = 0
        for i in np.arange(pixels_between_ends):
            edge_value_i = edge_dil[index[i, 0], index[i, 1]]
            sum_edge_value = sum_edge_value + edge_value_i
        avr_edge_value = np.floor((sum_edge_value / pixels_between_ends) * 100).astype('uint8')
    else:
        point = np.abs(e1[0] - e2[0]).astype(int)
        f = interpolate.interp1d(y, x)
        Y = np.linspace(e1[0], e2[0], num=point + 1, endpoint=True).astype(int)
        X = f(Y)
        Xr = np.round(X).astype(int)
        index = np.stack([Y, Xr], axis=1)
        index = index[~np.isnan(index.any(axis=1)), :]

        # Get average edge number
        pixels_between_ends = index.shape[0]
        sum_edge_value = 0
        for i in np.arange(pixels_between_ends):
            edge_value_i = edge_dil[index[i, 0], index[i, 1]]
            sum_edge_value = sum_edge_value + edge_value_i
        avr_edge_value = np.floor((sum_edge_value / pixels_between_ends) * 100).astype('uint8')
    return avr_edge_value, index

def get_lsp(skeleton1):
    if np.count_nonzero(skeleton1 == 1) > 10:
        skeleton1_obj = Skeleton(skeleton1)  # Get skeleton object
        df1 = summarize(skeleton1_obj, find_main_branch=True)  # Get dataframe
        main_idx = df1.index.values[(df1['main'] == True)]
        skeleton0 = np.zeros_like(skeleton1)
        for i in np.arange(np.size(main_idx)):
            main_coordinates = Skeleton.path_coordinates(skeleton1_obj, main_idx[i]).astype(
                'uint16')  # Get path coordinates
            for j in np.arange(main_coordinates.shape[0]):
                skeleton0[main_coordinates[j][0]][main_coordinates[j][1]] = 1
        # Dilate edge
        dilation_repeats = 4
        kernel_size = 3
        disk_dil = disk(kernel_size)
        skeleton0_uint8 = skeleton0.astype('uint8')
        skeleton0 = cv2.dilate(skeleton0_uint8, disk_dil, iterations=dilation_repeats)
        skeleton0 = skeletonize(skeleton0).astype(np.uint8)

    return skeleton0

def remove_j2e(skeleton, max_branch_size = 10000):
    skeleton2 = np.copy(skeleton)
    if np.count_nonzero(skeleton == 1) > 10:
        skeleton_obj = Skeleton(skeleton) # Get skeleton object
        df = summarize(skeleton_obj) # Get dataframe
        j2e_idx = df.index.values[(df['branch-type'] == 1)]  # Get id of junction-to-endpoint path
        for i, i_j2e_idx in enumerate(j2e_idx):
            if df['branch-distance'].values[i_j2e_idx] < max_branch_size: # Max size of pruning branch
                j2e_coordinates = Skeleton.path_coordinates(skeleton_obj, i_j2e_idx).astype('uint16') #Get path coordinates
                for j in np.arange(j2e_coordinates.shape[0]):
                    skeleton2[j2e_coordinates[j][0]][j2e_coordinates[j][1]] = 0  # Erase branches

    return skeleton2
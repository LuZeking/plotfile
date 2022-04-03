"""image processing
"""

from logging import error
import yaml
import pickle
import os
import glob
import cv2
import numpy as np
from analysis import setup_model_and_data
from analysis.setup_model_and_data import load_models
import tensorflow as tf

import PIL
from PIL import Image
from PIL import ImageShow
import datetime
import scipy.io
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import matplotlib.pyplot as plt

import json
from tqdm import tqdm

from skimage.transform import resize
from .load_anotations import  return_bbox_from_datasource, get_coco_id_name_map, get_ecoset_id2coco_id, return_bbox_01anno, return_bbox_from_datasource


def process_to_3x3_1_images(data_source, image_dir, image_filename, resize_min_size, format_suffix=None, to_3x3_1_images = False, save_moved_mask =False, class_name = False): #remove this function and import from imgs-to-hdf5
    '''Load image from path and preprocess for HDF5
    temporary copy from dist-TF2.
    '''
     # 101 
    if class_name:
        N = 3
    else:
        N = 101

    if format_suffix is None:
        # load image and convert to RGB
        image = PIL.Image.open(image_dir + image_filename) 
        #image.show()
        
    else:
        # should padding 0 
        img_id, index_anno = image_filename.split("_")
        zfill_img_id = img_id.zfill(12)
        image_filename = zfill_img_id + "_" + index_anno
        if os.path.exists(image_dir +image_filename + format_suffix) is True:
            #print("find typical")
            image = PIL.Image.open(image_dir +image_filename + format_suffix)
        # elif os.path.exists(image_dir +"atypical/"+image_filename + format_suffix) is True:
        #     # print("find atypical")
        #     image = PIL.Image.open(image_dir +"atypical/"+image_filename + format_suffix)
        # elif os.path.exists(image_dir +"fuzzy_region/"+image_filename + format_suffix) is True:
        #     # print("find fuzzy_region")
        #     image = PIL.Image.open(image_dir +"fuzzy_region/"+image_filename + format_suffix)
        else:
            # print(image_dir +"fuzzy_region/"+image_filename + format_suffix)
            print(f"something wrong, cannot find image!")
            # raise None# IOError
            if to_3x3_1_images:
                return None, np.zeros((N,131, 131,3)), None  # when no image found
            else: 
                return np.zeros((131, 131,3)), None
    # annotation_list = os.listdir("/home/hpczeji1/Datasets/annotations/coco_anotatation_txt/")
    # img_id, index_anno = image_filename.split("_")
    # zfill_img_id = img_id.zfill(12)
    
    # print(f"to_3x3_1_images {to_3x3_1_images}")
    if to_3x3_1_images:

        image_list_3x3_1 = np.zeros((N, 131, 131, 3)) # position_nums
        if save_moved_mask is True:
            moved_mask_3x3_1 = np.zeros((N, 131, 131))
        # image_list_3x3_1[0] = image

        # box = annotation['bbox']
        img_id, index_anno = image_filename.split("_")
        
        if save_moved_mask is True:
            box, mask_single = return_bbox_from_datasource(img_id, int(index_anno), data_source, save_moved_mask =False)
        else:
            box,_ = return_bbox_from_datasource(img_id, int(index_anno), data_source)# return_bbox_01anno(image_filename) #0-1 #! core2 : read txt and get bbox annotation
        if box is None: 
            print(f"read filename {image_filename} without bbox annotation!") #Todo fix the problem that without bbox 
            return None, np.zeros((N,131, 131,3)) 
        try:
            box = [float(i) for i in box]
        except:
            print(f"box: {box} here in wrong format")
            raise error
        left, top, right, bottom = box[0], box[1], box[0]+box[2], box[1]+box[3]
        
        
        image_copy = image.copy()
        # im_crop = image_copy((left, top, right, bottom))# crop 2  load annotation
        im_crop = image.crop((left, top, right, bottom)) 

        for k in range(0,N):
            if k == N-1:
                img_canvas = image
                if save_moved_mask is True:
                    moved_mask = resize(mask_single, (131,131))
            else:
                if class_name:
                    img_canvas=Image.new('RGB',(image_copy.width, image_copy.height),(255,255,255))  # "WHITE BACKGROUND"
                    cat_max_min_pos_dict = {1: [(53, 53), (0, 0)], 2: [(78, 38), (0, 0)], 3: [(65, 9), (0, 0)], 4: [(69, 57), (0, 0)], 5: [(58, 52), (0, 0)], 6: [(57, 47), (0, 0)], 7: [(57, 54), (0, 0)], 8: [(63, 53), (0, 0)], 9: [(72, 53), (0, 99)], 10: [(41, 50), (94, 0)], 14: [(71, 69), (0, 0)], 15: [(55, 52), (0, 0)], 16: [(57, 54), (0, 0)], 17: [(57, 55), (0, 0)], 18: [(59, 52), (0, 15)], 19: [(61, 56), (0, 0)], 20: [(59, 52), (0, 0)], 21: [(56, 52), (0, 0)], 22: [(57, 52), (0, 0)], 23: [(53, 52), (0, 0)], 24: [(52, 53), (0, 0)], 25: [(70, 36), (0, 0)], 26: [(24, 53), (99, 0)], 27: [(70, 46), (0, 0)], 33: [(92, 48), (0, 0)], 34: [(52, 52), (0, 0)], 40: [(33, 84), (0, 0)], 41: [(59, 54), (0, 0)], 42: [(23, 77), (0, 0)], 44: [(85, 90), (0, 0)], 45: [(68, 91), (0, 4)], 46: [(75, 47), (0, 0)], 47: [(62, 57), (0, 0)], 48: [(62, 60), (0, 0)], 51: [(64, 56), (0, 0)], 52: [(71, 49), (0, 14)], 53: [(70, 47), (0, 0)], 54: [(71, 49), (0, 0)], 55: [(64, 58), (0, 0)], 56: [(73, 50), (0, 0)], 57: [(75, 12), (0, 0)], 58: [(89, 7), (0, 0)], 60: [(93, 47), (0, 0)], 61: [(97, 49), (0, 0)], 62: [(74, 58), (0, 0)], 63: [(38, 52), (99, 0)], 64: [(77, 35), (0, 0)], 65: [(76, 82), (1, 16)], 68: [(58, 58), (0, 0)], 70: [(76, 47), (0, 0)], 71: [(54, 27), (99, 0)], 72: [(80, 30), (0, 0)], 73: [(47, 79), (0, 0)], 74: [(56, 24), (0, 0)], 75: [(49, 55), (99, 0)], 76: [(81, 56), (0, 0)], 77: [(58, 55), (0, 67)]}

                    i, j = cat_max_min_pos_dict[int(class_name)][k]
                    i, j = i//10, j//10


                else:
                    img_canvas=Image.new('RGB',(image_copy.width, image_copy.height),(255,255,255))  # "WHITE BACKGROUND"
                    j = k//10
                    i = k%10
                    # int(image_copy.width*(i*(1/10))), 
                    # x_paste, y_paste = int((image_copy.width)*((i)*(1/10))), int((image_copy.height)*((j)*(1/10)))
                x_paste, y_paste = int((image_copy.width-box[2])*((i)*(1/9))), int((image_copy.height-box[3])*((j)*(1/9)))
                # x, y = int(image_copy.width - box[2]), int(image_copy.height-box[3])# 0, 0 #! just show exsamples 
                # print(f"{(i,j)} paste in {(x_paste, y_paste)}, while image size is {(image_copy.width, image_copy.height)} ")
                # if (box[2]/2)>x: x = 0 # 填充超过边境
                # if (box[3]/2)>y: y = 0 #okTODO After GPU: not only fix the left, top edge, but bottom right also need to be fixed, or 3,6,7,8,9 will suffer from that
                # if x+(box[2]/2)>image_copy.width: x = int(image_copy.width-(box[2]/2))
                # if y+(box[3]/2)>image_copy.height: y = int(image_copy.height-(box[3]/2))
                img_canvas.paste(im_crop, (x_paste, y_paste))  # core 1 # calculate by i,j
                # print(f"x,y {(x_paste, y_paste)}")
                if save_moved_mask is True:
                    moved_mask = np.zeros(mask_single.shape)
                    moved_mask[ y_paste : y_paste+int(box[3]), x_paste:x_paste+int(box[2])] = mask_single[int(top) : int(top)+int(box[3]),int(left):int(left)+int(box[2])]
                    moved_mask = resize(moved_mask, (131,131))
     
            # copy from temp

            # following is the no
            img_canvas = img_canvas.convert('RGB')
            # reshape the image so the minimum dimension is at most resize_min_size
            image_min_side = min(img_canvas.size)
            if image_min_side > resize_min_size:
                # image larger than minimum size rescale
                image_resize_scale = image_min_side / resize_min_size
                scaled_size = (
                    round(img_canvas.width / image_resize_scale),
                    round(img_canvas.height / image_resize_scale))

                img_canvas = img_canvas.resize((131, 131)) # scaled_size
            else:
                img_canvas = img_canvas.resize((131, 131))

            # convert image to numpy array
            # and standardize to range [-1, 1]
            img_canvas = np.array(img_canvas)/(256/2)-1

            image_list_3x3_1[k] = img_canvas  # core 3 how to use it
            if save_moved_mask is True:
                moved_mask_3x3_1[k] = moved_mask


        if save_moved_mask is True:
            return None, image_list_3x3_1, moved_mask_3x3_1 #! need to change 
        return None, image_list_3x3_1, None
    else:
        image = image.convert('RGB')
        # reshape the image so the minimum dimension is at most resize_min_size
        image_min_side = min(image.size)
        if image_min_side > resize_min_size:
            # image larger than minimum size rescale
            image_resize_scale = image_min_side / resize_min_size
            scaled_size = (
                round(image.width / image_resize_scale),
                round(image.height / image_resize_scale))

            image = image.resize((131, 131)) # scaled_size
        else:
            image = image.resize((131, 131))




        # convert image to numpy array
        # and standardize to range [-1, 1]
        image = np.array(image)/(256/2)-1
        # 131, 131, 3

        return image, None

def discared_process_to_3x3_1_images(data_source, image_dir, image_filename, resize_min_size, format_suffix=None, to_3x3_1_images = False): #remove this function and import from imgs-to-hdf5
    '''Load image from path and preprocess for HDF5
    temporary copy from dist-TF2.
    '''
    if format_suffix is None:
        # load image and convert to RGB
        image = PIL.Image.open(image_dir + image_filename) 
        #image.show()
        
    else:
        # should padding 0 
        img_id, index_anno = image_filename.split("_")
        zfill_img_id = img_id.zfill(12)
        image_filename = zfill_img_id + "_" + index_anno
        if os.path.exists(image_dir +image_filename + format_suffix) is True:
            #print("find typical")
            image = PIL.Image.open(image_dir +image_filename + format_suffix)
        # elif os.path.exists(image_dir +"atypical/"+image_filename + format_suffix) is True:
        #     # print("find atypical")
        #     image = PIL.Image.open(image_dir +"atypical/"+image_filename + format_suffix)
        # elif os.path.exists(image_dir +"fuzzy_region/"+image_filename + format_suffix) is True:
        #     # print("find fuzzy_region")
        #     image = PIL.Image.open(image_dir +"fuzzy_region/"+image_filename + format_suffix)
        else:
            # print(image_dir +"fuzzy_region/"+image_filename + format_suffix)
            print(f"something wrong, cannot find image!")
            # raise None# IOError
            if to_3x3_1_images:
                return None, np.zeros((101,131, 131,3))  # when no image found
            else: 
                return np.zeros((131, 131,3)), None
    # annotation_list = os.listdir("/home/hpczeji1/Datasets/annotations/coco_anotatation_txt/")
    # img_id, index_anno = image_filename.split("_")
    # zfill_img_id = img_id.zfill(12)
    

    if to_3x3_1_images:

        image_list_3x3_1 = np.zeros((101, 131, 131, 3)) # position_nums
        # image_list_3x3_1[0] = image

        # box = annotation['bbox']
        img_id, index_anno = image_filename.split("_")
        box = return_bbox_from_datasource(img_id, int(index_anno), data_source)# return_bbox_01anno(image_filename) #0-1 #! core2 : read txt and get bbox annotation
        if box is None: 
            print(f"read filename {image_filename} without bbox annotation!") #Todo fix the problem that without bbox 
            return None, np.zeros((101,131, 131,3)) 
        box = [float(i) for i in box]
        left, top, right, bottom = box[0], box[1], box[0]+box[2], box[1]+box[3]
        
        
        image_copy = image.copy()
        # im_crop = image_copy((left, top, right, bottom))# crop 2  load annotation
        im_crop = image.crop((left, top, right, bottom)) 

        for k in range(0,101):
            if k == 0:
                img_canvas = image
            else:
                img_canvas=Image.new('RGB',(image_copy.width, image_copy.height),(255,255,255))  # "WHITE BACKGROUND"
                i = k//10
                j = k%10
                x, y = int(image_copy.width*(1/20+i*(1/10))), int(image_copy.height*(1/20+j*(1/10)))
                # if (box[2]/2)>x: x = 0 # 填充超过边境
                # if (box[3]/2)>y: y = 0 #okTODO After GPU: not only fix the left, top edge, but bottom right also need to be fixed, or 3,6,7,8,9 will suffer from that
                # if x+(box[2]/2)>image_copy.width: x = int(image_copy.width-(box[2]/2))
                # if y+(box[3]/2)>image_copy.height: x = int(image_copy.height-(box[3]/2))
                img_canvas.paste(im_crop, (x, y))  # core 1 # calculate by i,j
            # copy from temp

            # following is the no
            img_canvas = img_canvas.convert('RGB')
            # reshape the image so the minimum dimension is at most resize_min_size
            image_min_side = min(img_canvas.size)
            if image_min_side > resize_min_size:
                # image larger than minimum size rescale
                image_resize_scale = image_min_side / resize_min_size
                scaled_size = (
                    round(img_canvas.width / image_resize_scale),
                    round(img_canvas.height / image_resize_scale))

                img_canvas = img_canvas.resize((131, 131)) # scaled_size
            else:
                img_canvas = img_canvas.resize((131, 131))

            # convert image to numpy array
            # and standardize to range [-1, 1]
            img_canvas = np.array(img_canvas)/(256/2)-1

            image_list_3x3_1[k] = img_canvas  # core 3 how to use it


        return None, image_list_3x3_1 #! need to change 
    else:
        image = image.convert('RGB')
        # reshape the image so the minimum dimension is at most resize_min_size
        image_min_side = min(image.size)
        if image_min_side > resize_min_size:
            # image larger than minimum size rescale
            image_resize_scale = image_min_side / resize_min_size
            scaled_size = (
                round(image.width / image_resize_scale),
                round(image.height / image_resize_scale))

            image = image.resize((131, 131)) # scaled_size
        else:
            image = image.resize((131, 131))




        # convert image to numpy array
        # and standardize to range [-1, 1]
        image = np.array(image)/(256/2)-1
        # 131, 131, 3

        return image, None

def process_image(image_dir, image_filename, resize_min_size, format_suffix=None): #remove this function and import from imgs-to-hdf5
    '''Load image from path and preprocess for HDF5
    temporary copy from dist-TF2.
    '''
    if format_suffix is None:
        # load image and convert to RGB
        image = PIL.Image.open(image_dir + image_filename) 
        #image.show()
        
    else:
        # should padding 0 
        img_id, index_anno = image_filename.split("_")
        zfill_img_id = img_id.zfill(12)
        image_filename = zfill_img_id + "_" + index_anno
        try:
            image = PIL.Image.open(image_dir + image_filename+ format_suffix)
        except:
            if os.path.exists(image_dir +"typical/"+image_filename + format_suffix) is True:
                #print("find typical")
                image = PIL.Image.open(image_dir +"typical/"+image_filename + format_suffix)
            elif os.path.exists(image_dir +"atypical/"+image_filename + format_suffix) is True:
                # print("find atypical")
                image = PIL.Image.open(image_dir +"atypical/"+image_filename + format_suffix)
            elif os.path.exists(image_dir +"fuzzy_region/"+image_filename + format_suffix) is True:
                # print("find fuzzy_region")
                image = PIL.Image.open(image_dir +"fuzzy_region/"+image_filename + format_suffix)
            else:
                # print(image_dir +"fuzzy_region/"+image_filename + format_suffix)
                print(f" Shoud in {image_dir + image_filename+ format_suffix}, but no found!")
                # print(f"something wrong, cannot find image!")
                # raise None# IOError
                return np.zeros((131, 131,3))  # when no image found


    image = image.convert('RGB')


    # pad = False
    # if pad == True:
    #     image_arr = np.array(image)
    #     pad_l = int((resize_min_size * 25 / 12) / 2)
    #     image = np.pad(image_arr, ((pad_l, pad_l), (pad_l, pad_l), (0, 0)), mode='constant', constant_values=(0))

    #     image = Image.fromarray(np.uint8(image)).convert('RGB')



    # reshape the image so the minimum dimension is at most resize_min_size
    image_min_side = min(image.size)
    if image_min_side > resize_min_size:
        # image larger than minimum size rescale
        image_resize_scale = image_min_side / resize_min_size
        scaled_size = (
            round(image.width / image_resize_scale),
            round(image.height / image_resize_scale))

        image = image.resize((131, 131)) # scaled_size
    else:
        image = image.resize((131, 131))




    # convert image to numpy array
    # and standardize to range [-1, 1]
    image = np.array(image)/(256/2)-1
    # 131, 131, 3

    return image

def file_filter(f):
    if f.endswith("seg_with_background.png"):  # filter condition: seg_with_background.png is white , while cutseg.png is black
        return True
    else:
        return False
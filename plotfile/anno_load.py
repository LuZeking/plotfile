"""annotation loading
"""

from logging import error
import yaml
import pickle
import os
import glob
import cv2
import numpy as np
# from analysis import setup_model_and_data
# from analysis.setup_model_and_data import load_models
# import tensorflow as tf

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


def get_coco_id_name_map():
    coco_id_name_map ={ 1: 'person',
                  2: 'bicycle',
				  3: 'car',
				  4: 'motorcycle',
				  5: 'airplane',
                  6: 'bus',
				  7: 'train',
				  8: 'truck',
				  9: 'boat',
				  10: 'traffic light',
                  11: 'fire hydrant',
				  12: 'stop sign',
				  13: 'parking meter',
				  14: 'bench',
                  15: 'bird',
				  16: 'cat',
				  17: 'dog',
				  18: 'horse',
				  19: 'sheep',
				  20: 'cow',
                  21: 'elephant',
				  22: 'bear',
				  23: 'zebra',
				  24: 'giraffe',
				  25: 'backpack',
                  26: 'umbrella',
				  27: 'handbag',
				  28: 'tie',
				  29: 'suitcase',
				  30: 'frisbee',
                  31: 'skis',
				  32: 'snowboard',
				  33: 'sports ball',
				  34: 'kite',
				  35: 'baseball bat',
                  36: 'baseball glove',
				  37: 'skateboard',
				  38: 'surfboard',
				  39: 'tennis racket',
                  40: 'bottle',
				  41: 'wine glass',
				  42: 'cup',
				  43: 'fork',
				  44: 'knife',
				  45: 'spoon',
                  46: 'bowl',
				  47: 'banana',
				  48: 'apple',
				  49: 'sandwich',
				  50: 'orange',
                  51: 'broccoli',
				  52: 'carrot',
				  53: 'hot dog',
				  54: 'pizza',
				  55: 'donut',
                  56: 'cake',
				  57: 'chair',
				  58: 'couch',
				  59: 'potted plant',
				  60: 'bed',
				  61: 'dining table',
                  62: 'toilet',
				  63: 'tv',
				  64: 'laptop',
				  65: 'mouse',
				  66: 'remote',
				  67: 'keyboard',
                  68: 'cell phone',
				  69: 'microwave',
				  70: 'oven',
				  71: 'toaster',
				  72: 'sink',
                  73: 'refrigerator',
				  74: 'book',
				  75: 'clock',
				  76: 'vase',
				  77: 'scissors',
                  78: 'teddy bear',
				  79: 'hair drier',
				  80: 'toothbrush'}
    return  coco_id_name_map

def get_ecoset_id2coco_id():
    ecoset_id2coco_id = {1: 1, 2: None, 3: 3, 4: 1, 5: 68, 6: 60, 7: None, 8: 74, 9: 17, 10: 33, 11: None, 
    12: 18, 13: None, 14: None, 15: 1, 16: 9, 17: 61, 18: None, 19: 75, 20: 27, 21: None, 22: 42, 23: None, 
    24: None, 25: None, 26: 15, 27: None, 28: 6, 29: None, 30: 54, 31: 64, 32: None, 33: None, 34: None, 35: None,
    36: 48, 37: None, 38: None, 39: None, 40: None, 41: None, 42: None, 43: None, 44: 62, 45: 7, 46: 40, 47: None,
    48: None, 49: None, 50: None, 51: 16, 52: 8, 53: None, 54: 22, 55: None, 56: None, 57: None, 58: None, 59: None, 
    60: None, 61: 21, 62: 20, 63: 47, 64: None, 65: 44, 66: None, 67: None, 68: 73, 69: None, 70: None, 71: None, 
    72: None, 73: 26, 74: None, 75: None, 76: None, 77: None, 78: None, 79: None, 80: None, 81: None, 82: None, 
    83: None, 84: 76, 85: 52, 86: None, 87: None, 88: None, 89: None, 90: None, 91: None, 92: None, 93: None,
    94: 34, 95: None, 96: None, 97: None, 98: None, 99: None, 100: None, 101: None, 102: None, 103: None, 
    104: None, 105: None, 106: None, 107: None, 108: None, 109: None, 110: None, 111: None, 112: None, 113: None, 
    114: None, 115: 53, 116: None, 117: None, 118: None, 119: None, 120: None, 121: None, 122: None, 123: None, 124: None, 
    125: None, 126: None, 127: None, 128: None, 129: None, 130: None, 131: 4, 132: None, 133: None, 134: 5, 135: None, 136: None, 
    137: None, 138: None, 139: None, 140: 41, 141: None, 142: None, 143: None, 144: 45, 145: 70, 146: None, 147: None, 148: None, 149: None, 
    150: None, 151: None, 152: None, 153: None, 154: None, 155: None, 156: None, 157: None, 158: None, 159: None, 160: None, 
    161: None, 162: 55, 163: None, 164: None, 165: None, 166: 25, 167: None, 168: None, 169: None, 170: None, 171: None, 
    172: None, 173: None, 174: None, 175: None, 176: None, 177: None, 178: None, 179: None, 180: None, 181: None, 
    182: None, 183: None, 184: None, 185: None, 186: None, 187: None, 188: None, 189: None, 190: None, 191: None, 
    192: None, 193: None, 194: None, 195: None, 196: None, 197: None, 198: None, 199: None, 200: None, 201: None, 
    202: None, 203: None, 204: None, 205: None, 206: None, 207: None, 208: None, 209: None, 210: None, 211: None, 
    212: 19, 213: None, 214: None, 215: None, 216: None, 217: None, 218: None, 219: None, 220: None, 221: None,
        222: None, 223: None, 224: None, 225: None, 226: None, 227: None, 228: None, 229: None, 230: None, 231: None, 
        232: 56, 233: None, 234: None, 235: None, 236: None, 237: None, 238: 46, 239: None, 240: None, 241: None, 
        242: None, 243: None, 244: None, 245: 63, 246: None, 247: None, 248: None, 249: 10, 250: None, 251: None,
        252: None, 253: None, 254: None, 255: None, 256: 71, 257: None, 258: None, 259: 2, 260: None, 261: None,
        262: None, 263: None, 264: None, 265: None, 266: None, 267: None, 268: None, 269: None, 270: None, 271: None, 
        272: None, 273: None, 274: None, 275: None, 276: None, 277: None, 278: None, 279: 14, 280: None, 281: None, 
        282: None, 283: None, 284: None, 285: None, 286: None, 287: None, 288: None, 289: None, 290: None, 291: None, 
        292: None, 293: None, 294: None, 295: None, 296: None, 297: None, 298: None, 299: None, 300: None, 301: None, 
        302: None, 303: None, 304: None, 305: None, 306: 65, 307: None, 308: None, 309: None, 310: None, 311: None, 
        312: None, 313: None, 314: None, 315: None, 316: 51, 317: None, 318: None, 319: None, 320: None, 321: None, 
        322: None, 323: None, 324: None, 325: None, 326: None, 327: None, 328: None, 329: None, 330: None, 331: None, 
        332: None, 333: None, 334: 77, 335: None, 336: None, 337: 23, 338: None, 339: None, 340: None, 341: None,
        342: None, 343: None, 344: None, 345: None, 346: None, 347: None, 348: None, 349: None, 350: None, 351: None, 352: None, 353: None, 354: None, 355: None, 356: None, 357: None, 358: None, 359: None, 360: None, 361: None, 362: None, 363: None, 364: None, 365: None, 366: None, 367: None, 368: None, 369: None, 370: None, 371: None, 372: None, 373: None, 374: None, 375: None, 376: None, 377: None, 378: None, 379: None, 380: None, 381: None, 382: None, 383: None, 384: None, 385: None, 386: None, 387: None, 388: None, 389: None, 390: None, 391: None, 392: None, 393: None, 394: None, 395: None, 396: None, 397: None, 398: None, 399: None, 400: None, 401: None, 402: None, 403: None, 404: None, 405: None, 406: None, 407: None, 408: None, 409: None, 410: None, 411: None, 412: None, 413: None, 414: None, 415: None, 416: None, 417: None, 418: None, 419: None, 420: None, 421: None, 422: None, 423: None, 424: None, 425: None, 426: None, 427: None, 428: None, 429: None, 430: None, 431: None, 432: None, 433: None, 434: None, 435: None, 436: None, 437: None, 438: None, 439: None, 440: None, 441: None, 442: None, 443: None, 444: None, 445: None, 446: None, 447: None, 448: 72, 449: None, 450: None, 451: None, 452: None, 453: None, 454: None, 455: None, 456: None, 457: None, 458: None, 459: 58, 460: None, 461: None, 462: None, 463: None, 464: None, 465: None, 466: None, 467: None, 468: None, 469: None, 470: None, 471: None, 472: None, 473: None, 474: None, 475: None, 476: None, 477: None, 478: None, 479: None, 480: None, 481: None, 482: None, 483: None, 484: None, 485: None, 486: None, 487: None, 488: 24, 489: None, 490: None, 491: None, 492: None, 493: None, 494: None, 495: None, 496: None, 497: None, 498: 57, 499: None, 500: None, 501: None, 502: None, 503: None, 504: None, 505: None, 506: None, 507: None, 508: None, 509: None, 510: None, 511: None, 512: None, 513: None, 514: None, 515: None, 516: None, 517: None, 518: None, 519: None, 520: None, 521: None, 522: None, 523: None, 524: None, 525: None, 526: None, 527: None, 528: None, 529: None, 530: None, 531: None, 532: None, 533: None, 534: None, 535: None, 536: None, 537: None, 538: None, 539: None, 540: None, 541: None, 542: None, 543: None, 544: None, 545: None, 546: None, 547: None, 548: None, 549: None, 550: None, 551: None, 552: None, 553: None, 554: None, 555: None, 556: None, 557: None, 558: None, 559: None, 560: None, 561: None, 562: None, 563: None, 564: None, 565: None}


    return ecoset_id2coco_id

def return_bbox_01anno(image_filename):

    anno_dir = "/home/hpczeji1/Datasets/annotations/coco_anotatation_txt/"
    # annotation_list = os.listdir(anno_dir)

 


    img_id, index_anno = image_filename.split("_")
    zfill_img_id = img_id.zfill(12)
    f = open(anno_dir+zfill_img_id+".txt", "r")

    filename_anno_list = [i.strip() for i in f.readlines()]

    try:
        _, x, y, w, h, _, _, idx=  filename_anno_list[int(index_anno)].split()
        if int(idx) == int(index_anno):
            return (x, y, w, h)
        else:
            delat_idx = int(idx) - int(index_anno)
            _, x, y, w, h, _, _, idx=  filename_anno_list[int(index_anno)+delat_idx ].split()
            return (x, y, w, h)
    except:
        # delat_idx = int(idx) - int(index_anno)
        for i in range(int(index_anno), 1, -1):
            try:
                _, x, y, w, h, _, _, idx=  filename_anno_list[i].split()
                delat_idx = int(idx) - int(index_anno)
                # print(idx,index_anno, delat_idx)
                _, x, y, w, h, _, _, idx=  filename_anno_list[int(i)-delat_idx+1 ].split()
                # print(filename_anno_list[int(i)-delat_idx+1].split())
                return (x, y, w, h)
            except:
                continue

def return_bbox_from_datasource(img_id, idx, data_source, save_moved_mask = False):
    if isinstance(img_id, str):
        img_id = int(img_id.lstrip("0"))
    annotation_ids = data_source.getAnnIds(img_id)
    annotations = data_source.loadAnns(annotation_ids[idx]) 
    # print(f"anno {annotations}")

    if save_moved_mask:
        mask_single = data_source.annToMask(annotations[0])
        # mask_binary = mask_generator(data_source, 131, 131 [annotations[0]])
    else:
        mask_single = None
    return annotations[0]["bbox"], mask_single 

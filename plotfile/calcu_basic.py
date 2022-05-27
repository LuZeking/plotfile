from typing import List
import numpy as np
import time

def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes.

    Defined in :numref:`sec_anchor`"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def compute_d_prime(category_avg:List, category_var:List, non_category_avg:List, non_category_var:List):
    """D prime calculate the difference (in standard deviation units) between the means of  2 distribution，
        to measure the sensitivity or discriminability

    Args:
        category_avg (_type_): avg list of distribution_series1
        category_var (_type_): var list of distribution_series1
        non_category_avg (_type_):avg list of distribution_series2 
        non_category_var (_type_): avg list of distribution_series2

    Returns:
        D prime value
    """    
    nom = [category_avg[i] - non_category_avg[i] for i in np.arange(len(category_avg))]
    denom = [np.sqrt((category_var[i] + non_category_var[i]) / 2) for i in np.arange(len(category_avg))]
    with np.errstate(divide='ignore', invalid='ignore'):
        d_prime = [np.nan_to_num(nom[i] / denom[i]) for i in np.arange(len(category_avg))]
    return d_prime

class Timer:  #@save
    """record multi time """
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """start time"""
        self.tik = time.time()

    def stop(self):
        """stop and record in list"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """return avg time"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """return time same"""
        return sum(self.times)

    def cumsum(self):
        """return"""
        return np.array(self.times).cumsum().tolist()
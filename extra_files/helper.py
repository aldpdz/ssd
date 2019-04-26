
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from scipy import misc
from imageio import imwrite, imread


def save_images_to(path_from, path_to, list_img_names):
	'''
	Save images into a path
	path_from: path where to load the images
	path_to: path where to save the images
	list_img_names: list of string with the name's images
	'''
	for name_img in list_img_names:
		# Read image
		img = imread(path_from + '/' + name_img)
		imwrite(path_to + '/' + name_img, img)

def create_sub_dict(original_dict, list_keys):
	'''
	Creat a new dictionary from another one
	original_dict: original dictionary
	list_keys: list of string that containts the keys
	return: sub dictionary
	'''
	new_dict = {}
	for key in list_keys:
		new_dict[key] = original_dict[key]
	return new_dict


def load_images_labels(path, dict_json):
    '''
    Load images from a directory
    path: path to the image directory
    dict_json: json file
    return: images, labels as numpy arrays
    '''
    images = []
    labels = []
    for image in os.listdir(path):
        # Read image
        img = misc.imread(path + '/' + image)
        images.append(img)
        
        # Get data from json file
        b_boxes = get_ground_truth(dict_json[image])
        
        # Normilize bounding boxes
        b_boxes = normilize_boxes(b_boxes, img.shape[1], img.shape[0])
        labels.append(b_boxes)
    return np.array(images), np.array(labels)

def normilize_boxes(box_list, width, height):
    '''
    Convert boxes to float [0,1]
    box_list: list of bounding boxes
    width: original width
    height: original height
    '''
    normilize_list = []
    for box in box_list:
        new_box = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
        normilize_list.append(new_box)
    return np.array(normilize_list)

def get_ground_truth(list_gt):
    '''
    Return ground truth bounding boxes
    list_gt: dictionary that contains the boxes
    return: boxes as numpy arrays
    '''
    ground_truth_list = []
    for shape in list_gt:
        value = shape['shape_attributes']
        ground_truth = [value['x'], value['y'], value['width'], value['height']]
        ground_truth_list.append(ground_truth)
    return ground_truth_list

def clean_dict(json_file):
    '''
    Keep only useful information from the json file
    return: clean json file as a dictionary
    '''
    new_dict = {}
    for key, value in json_file.items():
        new_dict[value['filename']] = value['regions']
    return new_dict

def resize_images(list_image, width, height):
    '''
    Resize images from a numpy array
    list_image: numpy array of images
    width: new width
    height: new height
    return: numpy image
    '''
    return np.array([misc.imresize(img, size=(width, height)) for img in list_image])

def show_image_bb(img, bounding_boxs):
    '''
    Show image and its bounding boxes
    img: image
    bounding_box: list of size 4, x, y, width, height
    '''
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 12))
    # Display the image
    ax.imshow(img)

    for item in bounding_boxs:
        # Create a Rectangle patch
        if len(bounding_boxs[0]) > 4:
        	rect = patches.Rectangle((item[2], item[3]), item[4], item[5], linewidth=3, edgecolor='r', facecolor='none')
        else:
        	rect = patches.Rectangle((item[0], item[1]), item[2], item[3], linewidth=3, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        
    plt.show()

def show_image_bb_2(img, bounding_boxs_pred = [], bounding_boxs_gr = [], confidence = False):
    '''
    Show image and its bounding boxes
    img: image
    bounding_boxs_pred: bounding boxes from prediction x, y, width, height
    bounding_boxs_gr: ground true bounding boxes x, y, width, height, confidence
    '''
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 12))
    # Display the image
    ax.imshow(img)

    if len(bounding_boxs_pred) > 0:
        for item in bounding_boxs_pred:
            # Create a Rectangle patch
            rect = patches.Rectangle((item[0], item[1]), item[2], item[3], linewidth=3, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            if confidence:
                ax.text(item[0], item[1], round(item[4], 4), size='x-large', bbox=dict(facecolor='r', alpha=1))

    if len(bounding_boxs_gr) > 0:
        for item in bounding_boxs_gr:
            # Create a Rectangle patch
            rect = patches.Rectangle((item[0], item[1]), item[2], item[3], linewidth=3, edgecolor='b', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
        
    plt.show()

def normilize_to_pixel(list_elements, width, height):
    '''
    Convert float value to pixel
    list_elements: list of elements that contains the bounding boxes
    width: new width
    height: new height
    '''
    new_list = []
    for item in list_elements:
    	normilize_list = []
    	for box in item:
    		x = np.ceil(box[0] * width)
    		y = np.ceil(box[1] * height)
    		w = np.ceil(box[2] * width)
    		h = np.ceil(box[3] * height)
    		new_box = [x, y, w, h]
    		normilize_list.append(new_box)
    	new_list.append(normilize_list)
    
    return new_list


def get_batch(batch_size, features):
    """
    Send batch
    :param batch_size, tamaño del batch
    :param features: lista de imagenes
    :yield: batches de features
    """
    for start_i in range(0, len(features), batch_size):
        end_i = start_i + batch_size
        yield features[start_i:end_i]
    
def clean_predictions(pred, id_class=15):
    '''
    Keep just prediction with id 15 (person)
    pred: list of predictions
    '''
    new_pred = []
    for p in pred:
        box = []
        for item in p:
            if item[0] == id_class:
                box.append(item)
        new_pred.append(box)
    return np.array(new_pred)

def adjust_predictions(pred):
    '''
    Ajust predictions
    pred: list of predictions
    '''
    new_pred = []
    for i in pred:
        item = []
        for box in i:
        	# Convert corners to width and height
            b = [box[0], box[1], box[2], box[3], box[4] - box[2], box[5] - box[3]]
            b = np.array(b)
            item.append(b)
        new_pred.append(item)
    return np.array(new_pred)

def bb_intersection_over_union(boxA, boxB):
    '''
    Calculate Intersection over union
    boxA: first bounding box
    boxB: second bounding box
    return: iou
    '''
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2] + boxA[0], boxB[2] + boxB[0])
    yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground truth areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

def get_coordinates(list_items):
    '''
    Get just the coordinates from the list
    list_items: numpy that contains the predicted bounding boxes
    return: list with just the coordinates of the bounding boxes
    '''
    new_list = []
    for item in list_items:
        new_item = []
        for box in item:
            new_item.append([box[2], box[3], box[4], box[5]])
        new_list.append(new_item)
    return new_list

def best_match(square, list_two):
    '''
    Finds the best match by itersection over union
    square: item to compare with all items in list_two
    list_two: list of bounding boxes
    return: best iou
    '''
    max_iou = 0
    for bb in list_two:
        iou = bb_intersection_over_union(square, bb)
        if iou > max_iou:
            max_iou = iou
    return max_iou

def cal_precision(to_eval, to_compare, iou):
    '''
    Calculate the precision
    to_eval: cal precision for this item
    to_compare: item to compare with
    return: precision
    '''
    sum_iou = 0
    number_detection = 0
    for index_pred in range(len(to_eval)):
        number_detection += len(to_eval[index_pred])
        for item_to_eval in to_eval[index_pred]:
            best_iou = best_match(item_to_eval, to_compare[index_pred])
            if iou != None:
                if best_iou > iou:
                    best_iou = 1
                else:
                    best_iou = 0
            sum_iou += best_iou
            
    if number_detection == 0:
        return 0.0
    return sum_iou / number_detection

def cal_performance(ground_t, pred, verborse=True, iou=None):
    '''
    Calculate presicion, recall and F1 score
    ground_t: Bounding boxes of ground_t
    pred: Bounding boxes of prediction
    return: Presicion, Recall and F1 score
    '''
    presicion = cal_precision(pred, ground_t, iou)
    recall = cal_precision(ground_t, pred, iou)

    if (presicion + recall) > 0:
        f1_score = (2 * presicion * recall) / (presicion + recall)
    else:
    	f1_score = 0.0

    if(verborse):
    	print('Number of images:', len(ground_t))
    	print('Presicion:', round(presicion, 4))
    	print('Recall:', round(recall, 4))
    	print('F1 score:', round(f1_score, 4))
    
    return presicion, recall, f1_score

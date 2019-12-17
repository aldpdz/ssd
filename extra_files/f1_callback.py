import keras
import extra_files.helper as hp
import numpy as np
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
import pandas as pd
from imageio import imread
from skimage.transform import resize
from scipy import misc


class F1_callback(keras.callbacks.Callback):

    def __init__(self, confidence, iou, top_k, normalize_coords, height, width, output_shape, path_csv, path_save, data=None, label=None, label_csv=None, path_img=None, verborse=False,
                iou_f1=None):
        super(F1_callback, self).__init__()
        self.confidence = confidence
        self.iou = iou
        self.top_k = top_k
        self.normalize_coords = normalize_coords
        self.height = height
        self.width = width
        self.best_f1 = float("-inf")
        self.data = data
        self.label = label
        self.output_shape = output_shape
        self.path_csv = path_csv
        self.path_save = path_save
        self.history = []
        self.label_csv = label_csv
        self.path_img = path_img
        self.verborse = verborse
        self.iou_f1 = iou_f1

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        # Compute f1 score by applying nms
        
        # Make predictions
        # Create variable to store predictions
        predictions = np.zeros(shape=self.output_shape)
        if self.label_csv == None:
            for batch in hp.get_batch(32, self.data):
                pred = self.model.predict(batch)
                predictions = np.append(predictions, pred, axis=0)
        else:
            file_label = pd.read_csv(self.label_csv)
            # get all images' names
            file_column = file_label.columns
            img_val = file_label[file_column[0]].unique()

            normalized_label = []

            # Iterate over images
            for start_i in range(0, len(img_val), 32):
                end_i = start_i + 32
                input_ = []
                for img_name in img_val[start_i:end_i]:
                    img = imread(self.path_img + '/' + img_name)
                    height = img.shape[0]
                    width = img.shape[1]

                    # get labels from image
                    original_label = file_label[file_label[file_column[0]] == img_name].values[:, 1:-1]

                    # change formato from xmin, xmax, ymin, ymax to x, y, width, height
                    new_label = []
                    for o_label in original_label:
                        new_label.append([o_label[0], o_label[2], o_label[1] - o_label[0], o_label[3]- o_label[2]])

                    # normalized between [0,1]
                    new_label = hp.normilize_boxes(new_label, width, height)
                    normalized_label.append(new_label)

                    # resize image
                    resized_img= misc.imresize(img, size=(300, 300))
                    input_.append(resized_img)
                input_ = np.array(input_)
                input_ = input_.reshape(-1, 300, 300, 3)
                pred = self.model.predict(input_)
                predictions = np.append(predictions, pred, axis=0)

        predictions = predictions[1:] # delete empty item
                    
        # Decode predictinos
        pred_decod = decode_detections(predictions,
                                       confidence_thresh=self.confidence,
                                       iou_threshold=self.iou,
                                       top_k=self.top_k,
                                       normalize_coords=self.normalize_coords,
                                       img_height=self.height,
                                       img_width=self.width)
        
        pred_decod = np.array(pred_decod)
            
        # Remove class and confidence from predictions
        pred_decod = hp.clean_predictions(pred_decod, id_class=1)
        pred_decod = hp.adjust_predictions(pred_decod)
        pred_decod = hp.get_coordinates(pred_decod)
        
        aux_decod = []
        for item in pred_decod:
            aux_decod.append(hp.normilize_boxes(item, self.width, self.height))
        pred_decod = aux_decod

        # Calculate performance
        if self.label_csv == None:
            presicion, recall, f1_score = hp.cal_performance(self.label, pred_decod, verborse=self.verborse, iou=self.iou_f1)
        else:
            presicion, recall, f1_score = hp.cal_performance(normalized_label, pred_decod, verborse=self.verborse, iou=self.iou_f1)
        print('F1 score:', f1_score)
        
        self.history.append([epoch, presicion, recall, f1_score])
        
        # save file
        history_f1 = pd.DataFrame(self.history, columns=['epoch', 'presicion', 'recall', 'f1 score'])
        history_f1.to_csv(self.path_csv, index=False)
                
        if f1_score > self.best_f1:
            # Save model
            print('Improve F1 score from', self.best_f1, 'to', f1_score)
            self.best_f1 = f1_score
            self.model.save(self.path_save)

import keras
import extra_files.helper as hp
import numpy as np
from ssd_encoder_decoder.ssd_output_decoder import decode_detections

class F1_callback(keras.callbacks.Callback):

    def __init__(self, confidence, iou, top_k, normalize_coords, height, width, data, label):
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

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        # Compute f1 score by applying nms
        
        # Make predictions
        # Create variable to store predictions
        predictions = np.zeros(shape=(1, 2268, 14))
    
        for batch in hp.get_batch(32, self.data):
            pred = self.model.predict(batch)
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
        presicion, recall, f1_score = hp.cal_performance(self.label, pred_decod)
        
        if f1_score > self.best_f1:
            # Save model
            print('Improve F1 score from', self.best_f1, 'to', f1_score)
            self.best_f1 = f1_score

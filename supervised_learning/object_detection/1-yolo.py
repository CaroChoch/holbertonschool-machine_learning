#!/usr/bin/env python3#!/usr/bin/env python3
""" Class Yolo that uses the Yolo v3 algorithm to perform object detection """

import tensorflow as tf
import numpy as np
import tensorflow.keras as K
from tensorflow import keras as K


class Yolo:
    """
    Class Yolo that uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor
        Arguments:
            - model_path is the path to where a Darknet Keras model is stored
            - classes_path is the path to where the list of class names used
            for the Darknet model, listed in order of index, can be found
            - class_t is a float representing the box score threshold for
            the initial filtering step
            - nms_t is a float representing the IOU threshold for non-max
            suppression
            - anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes:
                - outputs is the number of outputs (predictions) made by
                the Darknet model
                - anchor_boxes is the number of anchor boxes used for each
                prediction
                - 2 => [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process outputs from Darknet model.

        Args:
            outputs (list): List of numpy.ndarrays containing predictions
                          Shape (grid_h, grid_w, anchors, 4 + 1 + classes)
                          4: (t_x, t_y, t_w, t_h)
                          1: box_confidence
                          classes: class probabilities
            image_size (ndarray): Original image size [height, width]

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
                  boxes: processed boundary boxes (x1, y1, x2, y2)
                  box_confidences: box confidences for each output
                  box_class_probs: class probabilities for each output
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            # Get current output dimensions
            grid_h, grid_w, anchors_count, _ = output.shape

            # Extract box components
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Get confidence scores and class probabilities
            box_conf = 1 / (1 + np.exp(-(output[..., 4:5])))
            class_prob = 1 / (1 + np.exp(-(output[..., 5:])))

            # Get model input dimensions
            input_w = self.model.input.shape[1].value
            input_h = self.model.input.shape[2].value

            # Get current anchor boxes
            current_anchors = self.anchors[i]
            anchor_w = current_anchors[:, 0]
            anchor_h = current_anchors[:, 1]

            # Reshape anchors for broadcasting
            box_w = anchor_w.reshape(1, 1, len(anchor_w))
            box_h = anchor_h.reshape(1, 1, len(anchor_h))

            # Calculate box center coordinates
            box_x = (t_x + grid_w / 2) / grid_w
            box_y = (t_y + grid_h / 2) / grid_h

            # Scale dimensions
            box_w = box_w * input_w
            box_h = box_h * input_h

            # Calculate box corner coordinates
            x1 = box_x - box_w / 2
            y1 = box_y - box_h / 2
            x2 = box_x + box_w / 2
            y2 = box_y + box_h / 2

            # Store processed outputs
            boxes.append(np.concatenate((x1, y1, x2, y2), axis=-1))
            box_confidences.append(box_conf)
            box_class_probs.append(class_prob)

        # Convert lists to numpy arrays
        boxes = np.array(boxes)
        box_confidences = np.array(box_confidences)
        box_class_probs = np.array(box_class_probs)

        # Return processed outputs : boxes, box_confidences, box_class_probs
        return boxes, box_confidences, box_class_probs

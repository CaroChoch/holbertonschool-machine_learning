#!/usr/bin/env python3
""" Class Yolo that uses the Yolo v3 algorithm to perform object detection """

import numpy as np
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
        """
        Converts model outputs into boundary box coordinates, confidences,
        and class probabilities.

        Args:
            - outputs (list of np.ndarray): Raw predictions from the model.
                Each array shape: (grid_h, grid_w,
                    num_anchors, 4 + 1 + num_classes)
            - image_size (tuple or list): Original image dimensions
                (height, width)

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
                - boxes: (x1, y1, x2, y2) coordinates in original image scale
                - box_confidences: objectness scores
                - box_class_probs: class probabilities
        """
        processed_boxes = []
        confidences = []
        class_probs = []

        img_h, img_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, num_anchors, _ = output.shape

            # Get box center coordinates using sigmoid
            box_xy = 1 / (1 + np.exp(-output[..., :2]))

            # Get box width and height using exponential and anchor boxes
            box_wh = np.exp(output[..., 2:4]) * self.anchors[i]

            # Confidence score for object presence
            object_confidence = 1 / (1 + np.exp(-output[..., 4:5]))

            # Class probabilities
            class_probabilities = 1 / (1 + np.exp(-output[..., 5:]))

            # Create grid indices for each cell
            grid_x = np.tile(np.arange(grid_w), grid_h).reshape(-1, grid_w)
            grid_y = np.tile(np.arange(grid_h).reshape(-1, 1), grid_w)

            # Reshape to match anchor dimensions
            grid_x = grid_x.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
            grid_y = grid_y.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)

            # Offset box center by grid cell position
            box_xy += np.concatenate((grid_x, grid_y), axis=-1)
            box_xy /= (grid_w, grid_h)
            box_wh /= (self.model.input.shape[1], self.model.input.shape[2])

            # Convert center coordinates to top-left and bottom-right
            top_left = box_xy - (box_wh / 2)
            top_left_scaled = top_left * (img_w, img_h)
            bottom_right_scaled = (top_left + box_wh) * (img_w, img_h)

            # Concatenate top-left and bottom-right points
            boxes = np.concatenate((top_left_scaled, bottom_right_scaled),
                                   axis=-1)

            # Append results for this output
            processed_boxes.append(boxes)
            confidences.append(object_confidence)
            class_probs.append(class_probabilities)

        return processed_boxes, confidences, class_probs

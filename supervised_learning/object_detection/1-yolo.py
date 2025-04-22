#!/usr/bin/env python3#!/usr/bin/env python3
""" Class Yolo that uses the Yolo v3 algorithm to perform object detection """

import tensorflow as tf
import numpy as np
import tensorflow.keras as K


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
        Process Darknet outputs
        Arguments:
            - outputs is a list of numpy.ndarrays containing the predictions
            from the Darknet model for a single image:
                - Each output will have the shape (grid_heighteight,
                grid_widthidth, anchor_boxes, 4 + 1 + classes)
                    - grid_heighteight & grid_widthidth => the height and
                    width of the grid used for the output
                    - anchor_boxes => the number of anchor boxes used
                    - 4 => (t_x, t_y, t_w, t_h)
                    - 1 => box_confidence
                    - classes => class probabilities for all classes
            - image_size is a numpy.ndarray containing the image’s original
            size [image_height, image_width]
        Returns:
            - tuple of (boxes, box_confidences, box_class_probs):
                - boxes: a list of numpy.ndarrays of shape (grid_heighteight,
                grid_widthidth, anchor_boxes, 4) containing the processed
                boundary boxes for each output, respectively:
                    - 4 => (x1, y1, x2, y2)
                - box_confidences: a list of numpy.ndarrays of shape
                (grid_heighteight, grid_widthidth, anchor_boxes, 1) containing
                the box confidences for each output, respectively
                - box_class_probs: a list of numpy.ndarrays of shape
                (grid_heighteight, grid_widthidth, anchor_boxes, classes)
                containing the class probabilities for each output,
                respectively
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i in range(len(outputs)):
            output = outputs[i]
            grid_height, grid_width, anchor_boxes, _ = output.shape
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]
            box_confidence = 1 / (1 + np.exp(-(output[..., 4:5])))
            box_class_prob = 1 / (1 + np.exp(-(output[..., 5:])))
            input_w = self.model.input.shape[1].value
            input_h = self.model.input.shape[2].value
            anchors = self.anchors[i]

            pw = anchors[:, 0]
            ph = anchors[:, 1]

            box_w = pw.reshape(1, 1, len(pw))
            box_h = ph.reshape(1, 1, len(ph))

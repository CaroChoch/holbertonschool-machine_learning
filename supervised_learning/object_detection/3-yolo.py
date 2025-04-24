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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on confidence score threshold and non-max
        suppression.

        Args:
            - boxes (list of np.ndarray): List of boundary boxes for each
            output
            - box_confidences (list of np.ndarray): List of box confidences for
            each output
            - box_class_probs (list of np.ndarray): List of box class
            probabilities for each output

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
                - filtered_boxes: (x1, y1, x2, y2) coordinates in original
                image scale
                - box_classes: class indices
                - box_scores: confidence scores
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            # Calculate box scores by multiplying confidences with class
            # probabilities
            box_scores_i = box_confidences[i] * box_class_probs[i]

            # Find the class with the highest score for each box
            box_classes_i = np.argmax(box_scores_i, axis=-1)
            box_class_scores_i = np.max(box_scores_i, axis=-1)

            # Create a mask for boxes with scores above the threshold
            mask = box_class_scores_i >= self.class_t

            # Apply mask to filter boxes, classes, and scores
            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_classes_i[mask])
            box_scores.append(box_class_scores_i[mask])

        # Concatenate results from all outputs
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-max suppression to filter out overlapping boxes.

        Args:
            - filtered_boxes (np.ndarray): Filtered bounding boxes
            - box_classes (np.ndarray): Class indices for each box
            - box_scores (np.ndarray): Confidence scores for each box

        Returns:
            - tuple: (box_predictions, predicted_box_classes,
            predicted_box_scores)
                - box_predictions: Predicted bounding boxes after NMS
                - predicted_box_classes: Class indices for predicted boxes
                - predicted_box_scores: Confidence scores for predicted boxes
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            # Get indices of boxes with the current class
            cls_indices = np.where(box_classes == cls)

            # Extract boxes, scores for the current class
            cls_boxes = filtered_boxes[cls_indices]
            cls_scores = box_scores[cls_indices]

            # Sort boxes by score in descending order
            sorted_indices = np.argsort(-cls_scores)
            cls_boxes = cls_boxes[sorted_indices]
            cls_scores = cls_scores[sorted_indices]

            while len(cls_boxes) > 0:
                # Select the box with the highest score
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[0])

                # Calculate IoU of the selected box with the rest
                ious = self.iou(cls_boxes[0], cls_boxes[1:])

                # Remove boxes with IoU above the threshold
                cls_boxes = cls_boxes[1:][ious < self.nms_t]
                cls_scores = cls_scores[1:][ious < self.nms_t]

        return (np.array(box_predictions),
                np.array(predicted_box_classes),
                np.array(predicted_box_scores))

    def iou(self, box1, boxes):
        """
        Calculate Intersection over Union (IoU) between a box and an array of
        boxes.

        Args:
            - box1 (np.ndarray): A single box
            - boxes (np.ndarray): Array of boxes

        Returns:
            np.ndarray: IoU values
        """
        # Calculate the (x, y) coordinates of the intersection rectangle
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])

        # Compute the area of intersection rectangle
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Compute the area of both the prediction and ground-truth rectangles
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Compute the union area by using the formula:
        # union(A,B) = A + B - inter(A,B)
        union_area = box1_area + boxes_area - inter_area

        # Compute the IoU
        return inter_area / union_area

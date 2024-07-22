import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import math

from ultralytics import YOLO



model = YOLO("E:/Combined-files/Combined-files/best.pt")

# Function to perform inference and get normalized bounding box coordinates in xywh format for each frame
def get_bounding_boxes(frame, img_width, img_height):
    # Perform inference
    results = model.predict(source=frame, save=False)

    # Extract bounding box coordinates in x_center, y_center, width, height format
    bounding_boxes = results[0].boxes.xywh  # [x_center, y_center, width, height]

    # Convert to numpy arraywhwww
    bounding_boxes = bounding_boxes.cpu().numpy()

    # Normalize the bounding box coordinates
    bounding_boxes[:, 0] /= img_width   # Normalize x_center
    bounding_boxes[:, 1] /= img_height  # Normalize y_center
    bounding_boxes[:, 2] /= img_width   # Normalize width
    bounding_boxes[:, 3] /= img_height  # Normalize height

    return bounding_boxes

#Function to visualize bounding boxes on a frame and print the coordinates
def visualize_boxes_xywh(mid_line_coord, frame, boxes, img_width, img_height):
    print("From left_right " , boxes)
    x_center_player, _, _, _ = boxes[0]
    if x_center_player < mid_line_coord:
        p1_side = "Left"
    else:
        p1_side = "Right"
    
    print("Player 1 is on the", p1_side)

    return frame, p1_side



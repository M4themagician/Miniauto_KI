"""Provides visualization utilities for supported tasks.
"""

import torch
import torchvision
import torchvision.models as models
from torchvision.utils import draw_bounding_boxes, draw_keypoints

import cv2
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np

class MplColorHelper:
    """TODO: Docstring"""
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.start_val = start_val
        self.stop_val = stop_val
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val, P):
        if val == 255: return (255, 255, 255, 1)
        if val < self.start_val or val > self.stop_val: return (0, 0, 0)
        color_float_rgba = self.scalarMap.to_rgba(val)
        color = (int(color_float_rgba[0]*255), int(color_float_rgba[1]*255), int(color_float_rgba[2]*255), int(255*P**4))
        return color

class vis():
    def __init__(self, resolution):
        self.resolution = resolution
        self.imagenet1k_names = models.ResNet34_Weights.DEFAULT.meta['categories']
        self.coco_names = models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT.meta['categories']
        self.coco_keypoint_names = models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT.meta['categories']
        self.seg_names = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT.meta['categories']
        
        # setup segmentation visualization
        r, g, b = np.zeros(256, dtype = np.dtype('uint8')), np.zeros(256, dtype = np.dtype('uint8')), np.zeros(256, dtype = np.dtype('uint8'))
        self.toImage = torchvision.transforms.ToPILImage()
        self.seg_cmap = MplColorHelper('turbo', 1, len(self.seg_names)+1)
        for k in range(len(self.seg_names)):
            rgb = self.seg_cmap.get_rgb(k, 1)
            r[k] = rgb[2]
            g[k] = rgb[1]
            b[k] = rgb[0]


        self.lut = np.dstack( (b, g, r) ).astype(np.dtype('uint8'))
            
    def visualize(self, type, image, input_tensor, prediction):
        match type:
            case "classification":
                return self.classification(image, prediction)
            case "segmentation":
                return self.segmentation(image, prediction)
            case "object_detection":
                resolution = input_tensor.size()[2:]
                return self.object_detection(image, prediction, resolution)
            case "keypoint":
                resolution = input_tensor.size()[2:]
                return self.keypoint(image, prediction, resolution)
        
        
    def classification(self, image, prediction, display_top_k = 5):
        # prediction is an N x 1000 logits tensor
        prediction = torch.softmax(prediction, dim=1)[0]
        topk = torch.topk(prediction, display_top_k, sorted=True)
        h, w = image.shape[:2]
        
        x_offset = 30
        y_offset = h - 250
        spacing = 40
        for k, value in enumerate(topk.values):
            cv2.putText(image, f"{int(value*100):02d}% {self.imagenet1k_names[topk.indices[k]]}", org = (int(x_offset), int((y_offset + k*spacing))), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
            
        return image
    
    def segmentation(self, image, prediction):
        # prediction is a dictionary, prediction['out'] is N x 21 x 520 x 924 tensor
        prediction = torch.argmax(prediction['out'], dim=1).to(torch.uint8).unsqueeze(1)
        prediction = torch.nn.functional.interpolate(prediction, size = (self.resolution[1], self.resolution[0]), mode='nearest').squeeze(1).cpu()
        image = cv2.resize(image, dsize=self.resolution)
        mask = prediction[0].to(torch.uint8).numpy()
        color_mask = cv2.LUT(np.dstack((mask,mask,mask)), self.lut)
        colored_pixels = np.isin(mask, [0], invert = False)
        color_mask[colored_pixels] = image[colored_pixels]
        return color_mask
    
    def object_detection(self, frame, predictions, input_resolution):
        # predictions: list (over batch_size) of dicts containing
        # 'boxes' : N x 4 tensor with absolute box coordinates
        # 'labels' : N tensor (int) with object indices
        # 'scores' : N tensor (float) with object scores
        
        score_thresh = 0.5

        frame_res = frame.shape[:2]
        boxes = predictions[0]['boxes'].cpu().mul(torch.tensor([frame_res[0]/input_resolution[0], frame_res[1]/input_resolution[1], frame_res[0]/input_resolution[0], frame_res[1]/input_resolution[1]]))
        scores = predictions[0]['scores'].cpu()
        boxes = boxes[scores > score_thresh]
        labels = [self.coco_names[k] for k in predictions[0]['labels'][scores > score_thresh]]
        tensor_image = torch.from_numpy(frame).to(torch.uint8).permute(2, 0, 1)
        vis = draw_bounding_boxes(tensor_image, boxes, labels, width=4).permute(1, 2, 0)
        return vis.numpy()
    
    def keypoint(self, frame, predictions, input_resolution):
        # predictions: list (over batch_size) of dicts containing
        # 'boxes' : N x 4 tensor with boxes in absolute image coordinates
        # 'labels' : N tensor (int) with object indices
        # 'scores' : N tensor (float) with object scores
        # 'keypoints' : N x 17 x 3 tensor containing keypoints (absolute image coordinates) 
        score_thresh = 0.5
        connect_skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (6, 8),
            (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        frame_res = frame.shape[:2]
        boxes = predictions[0]['boxes'].cpu().mul(torch.tensor([frame_res[0]/input_resolution[0], frame_res[1]/input_resolution[1], frame_res[0]/input_resolution[0], frame_res[1]/input_resolution[1]]))
        scores = predictions[0]['scores'].cpu()
        keypoints = predictions[0]['keypoints'].cpu()
        boxes = boxes[scores > score_thresh]
        labels = [self.coco_names[k] for k in predictions[0]['labels'][scores > score_thresh]]
        keypoints = keypoints[scores > score_thresh]
        tensor_image = torch.from_numpy(frame).to(torch.uint8).permute(2, 0, 1)
        vis = draw_keypoints(tensor_image, keypoints, connectivity=connect_skeleton, colors="blue", radius=4, width = 3).permute(1, 2, 0)
        return vis.numpy()
    
    
if __name__ == "__main__":
    visualizer = vis()
    
    
"""model.py should provide a class handling 
1. Opening a webcam and reading its images.
2. Instanciating models from the torchvision model zoo.
3. Passing the webcam images into the current model.
4. Visualizing the outputs by instanciating a vis object (see vis.py) and passing images and predictions to it.
5. Passing the visualized network outputs to the GUI.
6. Allow changing the model on the fly via some interface.
"""

import torchvision.models as models
import cv2
from PIL import Image

from backend.vis import vis






class ModelInferencer():
    def __init__(self, camera, initial_algo_type = 'classification', resolution = (960, 480), flip_rgb=True):
        self.loaded_algo_type = "none"
        self.algo_type = initial_algo_type
        self.resolution = resolution
        self.camera = camera
        self.model = None
        self.transform = None
        self.flip_rgb = flip_rgb
        
        self.visualizer = vis(resolution)
        
    def get_frame(self):
        # Load model if not initialized or algo_type changed:
        self.get_model()
        
        grabbed, frame = self.camera.read()
        assert grabbed, "Could not read from camera."
        
        pil_frame = Image.fromarray(frame)
        input_tensor = self.transform(pil_frame).cuda().unsqueeze(0)
        network_output = self.model(input_tensor)
        frame = self.visualizer.visualize(self.loaded_algo_type, frame, input_tensor, network_output)
        
        
        frame = self.resize_frame(frame)
        if self.flip_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    
    def set_webcam(self, id):
        pass
    
    def set_model(self, string_of_type: str):

        """Sets the current model to that chosen by string_of_type.
        Lazily loads the model into gpu memory

        Args:
            string_of_type (str): The type of model to load. Can be one of "classification", "object_detection", "segmentation", "keypoint_detection"
        """
        if string_of_type != "none":
            self.algo_type = string_of_type
    
    def get_model(self):
        if self.loaded_algo_type != self.algo_type:
            assert self.algo_type in ['none', 'classification', 'object_detection', 'segmentation', 'keypoint']
            match self.algo_type:
                case "classification":
                    weights = models.ResNet34_Weights.DEFAULT
                    self.model = models.resnet34(weights=weights).cuda()  
                case "object_detection":
                    weights = models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
                    self.model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights).cuda()
                case "segmentation":
                    weights = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
                    self.model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights).cuda()
                case "keypoint":
                    weights = models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
                    self.model = models.detection.keypointrcnn_resnet50_fpn(weights=weights).cuda()
            self.model.eval()
            self.transform = weights.transforms()
            self.loaded_algo_type = self.algo_type
            
    def resize_frame(self, frame):
        h, w = frame.shape[:2]
        t_h, t_w = self.resolution[1], self.resolution[0]
        if w/h > t_w/t_h:
            # input image is wider than what we want, use full height
            new_w = t_w/t_h*h
            x_off = (w - int(new_w))//2
            y_off = 0
            
        else:
            # we want a wider image than what's supplied, use full width and crop height:
            new_h = w/t_w*t_h
            y_off = (h - int(new_h))//2
            x_off = 0
        frame = frame[y_off:h-y_off, x_off:w-x_off, :] 
        frame = cv2.resize(frame, dsize=self.resolution)
        return frame
        
            
            
if __name__ == "__main__":
    path2testvideo = '/localdata/Miniauto_demo/000216_001.mp4'
    camera = cv2.VideoCapture(path2testvideo)
    
    network = ModelInferencer(camera, flip_rgb=False)
    
    while True:
        frame = network.get_frame()
        cv2.imshow('output', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    
            
                
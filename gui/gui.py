"""This gui should do some things:
1. Show a opencv-matrix / numpy array as an RGB (or BGR) image on a pane somewhere.
2. Have a column or row of buttons somewhere around the displayed image, to chose a network.
3. Each button should create some event when pressed, from which the desired model type can be inferred and chosen in the backend.
"""

from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication
from PyQt5 import QtGui, QtWidgets
import numpy as np
from pyqtgraph import ImageView
import cv2
from backend.model import ModelInferencer

class MiniautoGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KI-Tage Visualisierung")
        #webcam = cv2.VideoCapture(0)
        path2testvideo = '/localdata/Miniauto_demo/000216_001.mp4'
        webcam = cv2.VideoCapture(path2testvideo)
        self.model = ModelInferencer(webcam)
        self.central_widget = QWidget()
        self.button_Class = QPushButton('Classification', self.central_widget)
        self.button_Class.resize(100,32)
        self.button_OD = QPushButton('Object Detection', self.central_widget)
        self.button_Seg = QPushButton('Segmentation', self.central_widget)
        self.button_Key = QPushButton('Keypoint', self.central_widget)
        self.button_Stop = QPushButton('Stop', self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.button_Class)
        self.layout.addWidget(self.button_OD)
        self.layout.addWidget(self.button_Seg)
        self.layout.addWidget(self.button_Key)
        self.layout.addWidget(self.button_Stop)
        self.setCentralWidget(self.central_widget)
        self.image_view = QtWidgets.QLabel() #ImageView()
        self.layout.addWidget(self.image_view)

        self.button_Class.clicked.connect(self.change_model_to_Class)
        self.button_Class.clicked.connect(self.startTimer)
        self.button_OD.clicked.connect(self.change_model_to_OD)
        self.button_OD.clicked.connect(self.startTimer)
        self.button_Seg.clicked.connect(self.change_model_to_Seg)
        self.button_Seg.clicked.connect(self.startTimer)
        self.button_Key.clicked.connect(self.change_model_to_key)
        self.button_Key.clicked.connect(self.startTimer)
        self.button_Stop.clicked.connect(self.stop)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_image)

    def startTimer(self):
        self.update_timer.start(50)
        #self.startBtn.setEnabled(False)
        #self.endBtn.setEnabled(True)

    def update_image(self):
        frame = self.model.get_frame()
        frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        self.image_view.setPixmap(QtGui.QPixmap.fromImage(frame.scaled(1280, 720)))

    def change_model_to_Class(self):
        button = 'classification'
        self.model.set_model(button)

    def change_model_to_OD(self):
        button = 'object_detection'
        self.model.set_model(button)

    def change_model_to_Seg(self):
        button = 'segmentation'
        self.model.set_model(button)

    def change_model_to_key(self):
        button = 'keypoint'
        self.model.set_model(button)

    def stop(self):
        self.update_timer.stop()
        #self.app.exit(self.app.exec_())


if __name__ == '__main__':
    app = QApplication([])
    window = MiniautoGUI()
    window.show()
    app.exit(app.exec_())
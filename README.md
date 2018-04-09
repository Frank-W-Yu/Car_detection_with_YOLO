# Car_detection_with_YOLO

This is a project of Coursera Deep Learning Convolutional Neural Network course. According to the honor code of Coursera, the actual solution to the project is not uploaded.

YOLO ("you only look once") is a popular algorithm for object detection due to its high accuracy and relatively low computing cost. It allows to run the real-time oject detection with 45 frames per second, which is an ideal algorithm for the autopilot. This algorithm only requires one forward propagation passing through the network to make predictions that it outputs all the recoginized objects with their bounding boxes.

## YOLO model architecture
-------------------------
This model has 23 convolution layers including 1 residual network layer. The following schematic shows a brief abstract of the architecture. 
<img src="https://github.com/Frank-W-Yu/Car_detection_with_YOLO/blob/master/nb_images/model_architecture.png" style="width:500px;height:250;">

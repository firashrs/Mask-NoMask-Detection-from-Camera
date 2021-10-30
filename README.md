# Mask-NoMask-Detection-from-Camera
An <b>object detection</b> project, capable of identifying if a person is wearing a mask or not in <b>real-time</b>.<br/>
By running the project, the camera will be activated and the stream of pictures from it will be used as input for the Mask detection model.
The output will then be calculated and shown along with the live feed.<br>
## PART I: The TensorFlow Lite Model
For better performance on the <b>Raspberry Pi</b>, the model must be exported as <b>TensorFlow Lite</b>.<br/>
The <b>Jupyter</b> notebook file contains the code for training the object detection model. This model is based on the <b>MobileNetV2</b> network. 

## Part II: Running the model on a live Camera
To perform <b>inferencing</b> on the camera feed, the script uses <b>OpenCV</b> to read frames continuously.
The frames are then passed as inputs to the Mask detection model.
The outputs are either <b>mask or no mask</b>.

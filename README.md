# Deploy FC-DenseNet-103 model by OpenVINO™ Toolkit
## Introduction
  This article is to evaluate how the neural network topology of "FC-DenseNet-103" is implemented on Intel hardware to achieve low-power, low-cost and low-noise edge computing solutions in hospital operating rooms。
  The original paper source of the model RC-DenseNet-103 is https://arxiv.org/pdf/1611.09326.pdf
![image](https://user-images.githubusercontent.com/91500280/135002637-93a470bc-01be-4b4a-8dcd-99c0ed4328af.png)

This article is divided into four parts：
-	Install TensorFlow related support and verify the installation
-	Download and run FC-DenseNet-03 pre-trained model
-	Convert Keras h5 model to ONNX model and IR format file
-	Perform performance test on Intel target hardware by Benchmark_APP

Tested with：
-	OpenVINO™   2021.4 LTS
-	TensorFlow       2.2.0
-	Python          3.8.10

## 1.	TensorFlow installation guide 

**Step 1**：Create a virtual environment with Python=3.8 in Anaconda：
```Python
 Conda create -n tensorflow2_2 python=3.8
```
**Step 2**：Pip install tensorflow 2.2.0 
```Python
pip install –ignore-installed –upgrade tensorflow==2.2.0
```
**【Annotation】** Install TensorFlow 2.5.0 version, when using the convert_to_onnx.py script to convert the ONNX model, an error message of "AttributeError: ‘KerasTensor’ object has no attribute ‘graph’ will appear. It has been verified that the conversion can be successful using TensorFlow 2.2.0.

**Step 3**：Install cudatoolkit=10.1 cudnn=7.6.5
```Python
conda install cudatoolkit=10.1 cudnn=7.6.5
```
**【Annotation】** The video memory size of the device used in this article is 6GB. When using the pre-trained model for model training, there will be "insufficient video memory". I can use the TensorFlow-CPU version to complete the model training.

**Step 4**：Enter the code as shown in Code 1 to verify whether the TensorFlow installation is successful.
```Python
>python
>>>import tensorflow as tf
>>>hello = tf.constant('hello,tensorflow!')
>>>print(hello.device)
>>>print(hello)
```
If the result shown in Figure appears, it proves that TensorFlow 2.2.0 is installed successfully.

![image](https://user-images.githubusercontent.com/91500280/135003704-9de75dfc-5e13-4881-b0ea-4a511690893a.png)

## 2.Running a FC-DenseNet-03 demo
**Step 1**: Clone the source code to a local folder
```Python
git clone https://github.com/lukszamarcin/100-tiramisu-keras
```
**Step 2**: The TensorFlow pre-training model used in this article.Put the downloaded Keras h5 format model into the models folder in the source code
```
https://github.com/lukszamarcin/100-tiramisu-keras
```
**Step 3**: Test the network trained with Camvid data on a custom image
```Python
python run_tiramisu_camvid.py
```

## 3.	Convert model into ONNX and IR format
**Step 1**: Install the latest version of Keras2ONNX from the source code：
```Python
pip install keras2onnx
```
The command to install from source code is：
```Python
pip install -U git+https://github.com/microsoft/onnxconverter-common
pip install -U git+https://github.com/onnx/keras-onnx
```
**【Annotation】** Installation keras2onnx refer to:  https://pypi.org/project/keras2onnx/

**Step 2**: Running the convert_to_onnx.py Python script to convert the Keras h5 model to the ONNX model.
```Python
python convert_to_onnx.py
```
The specific content of the script is as follows:
```Python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tiramisu.model import create_tiramisu
import keras2onnx
# Set the weight file name
keras_model_weights = "models/my_tiramisu.h5"
onnx_model_weights = keras_model_weights.split('.')[0]+'.onnx'
# Load model and weights
input_shape = (224, 224, 3)
number_classes = 32  # CamVid data consist of 32 classes
# Prepare the model information
img_input = Input(shape=input_shape, batch_size=1)
x = create_tiramisu(number_classes, img_input)
model = Model(img_input, x)
# Load the keras model weights
model.load_weights(keras_model_weights)
print("Line17")
print(model.name)
onnx_model = keras2onnx.convert_keras(model, model.name)
# Save the onnx model weights
keras2onnx.save_model(onnx_model, onnx_model_weights)
```

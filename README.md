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


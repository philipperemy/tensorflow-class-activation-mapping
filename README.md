# Learning Deep Features for Discriminative Localization
**Tensorflow Implementation**

https://arxiv.org/abs/1512.04150

## How to use it?

```
# For Python 3. Should also work for Python2 :)
git clone git@github.com:philipperemy/tensorflow-class-activation-mapping.git
cd tensorflow-class-activation-mapping
pip3 install -r requirements.txt
mkdir /tmp/img
tar xvzf data/mnist_cluttered.tar.gz -C /tmp/img
python3 mnist.py # The heatmaps are available in out/
```

## Abstract

*In this work, we revisit the global average pooling layer and shed light on how it explicitly enables the convolutional neural network to have remarkable localization ability despite being trained on image-level labels. While this technique was previously proposed as a means for regularizing training, we find that it actually builds a generic localizable deep representation that can be applied to a variety of tasks. Despite the apparent simplicity of global average pooling, we are able to achieve 37.1% top-5 error for object localization on ILSVRC 2014, which is remarkably close to the 34.2% top-5 error achieved by a fully supervised CNN approach. We demonstrate that our network is able to localize the discriminative image regions on a variety of tasks despite not being trained for them.*


## Class Activation Mapping and Class-specific Saliency Map

*We propose a technique for generating class activation maps using the global average pooling (GAP) in CNNs. A class activation map for a particular category indicates the discriminative image regions used by the CNN to identify that category. The procedure for generating these maps is illustrated as follows:*

<div align="center">
  <img src="http://cnnlocalization.csail.mit.edu/framework.jpg"><br><br>
</div>

*Class activation maps could be used to intepret the prediction decision made by the CNN. The left image below shows the class activation map of top 5 predictions respectively, you can see that the CNN is triggered by different semantic regions of the image for different predictions. The right image below shows the CNN learns to localize the common visual patterns for the same object class.*

<div align="center">
  <img src="http://cnnlocalization.csail.mit.edu/example.jpg"><br><br>
</div>



## MNIST cluttered dataset with LeNet

*LeNet* was used on the [MNIST cluttered dataset]((https://github.com/deepmind/mnist-cluttered)), provided by DeepMind.
We removed the last fully connected layers and replaced them with the **Class Activation Map**.
The accuracy was **97%** after 50 epochs (Final accuracy was **97.6%**). Here is an example of an input image (9):
<div align="center">
  <img src="img/image_test.png"><br><br>
</div>

The output of the **Class Activation Map** is provided below, at different steps:
<div align="center">
  <img src="img/result.png"><br><br>
</div>

The overall accuracy on the testing set, per step:
<div align="center">
  <img src="img/accuracy_mnist_cluttered.png" width="400px"><br><br>
</div>



## CALTECH256 dataset with AlexNet pretrained with ImageNet

Coming soon!

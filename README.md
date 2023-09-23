# Convolution Model from Scratch

This repository is the implementation of convolutional(CONV) and pooling (POOL) layers in numpy, including both forward and backward propagation.

## Convolutional Neural Networks

A convolution layer transforms an input volume into an output volume of different size, as shown below. 

<img src="images/conv_nn.png" style="width:350px;height:200px;">

### Building block of the convolutional neural network

- Convolution functions, including:
    * Zero Padding
    * Convolve window 
    * Convolution forward
    * Convolution backward
- Pooling functions, including:
    * Pooling forward
    * Create mask 
    * Distribute value
    * Pooling backward

#### Architecture

<img src="images/model.png" style="width:900px;height:600px;">

##### Zero Padding
<img src="images/PAD.png" style="width:600px;height:400px;">

##### Single Step Convolution
<img src="images/Convolution_schematic.gif" style="width:900px;height:600px;">

##### Convolutional Neural Networks - Forward Pass


##### Pooling Layer

<table>
<td>
<img src="images/max_pool1.png" style="width:500px;height:300px;">
<td>

<td>
<img src="images/a_pool.png" style="width:500px;height:300px;">
<td>
</table>


## Setup
1. Clone this repository

```shell
git clone https://github.com/abel-shimeles/CNN_From_Scratch
cd CNN_From_Scratch
```

2. Using the cnn.py file build your own CNN Model by using your own data.




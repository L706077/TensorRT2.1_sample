
## Download
- [TensorRT2.1](https://developer.nvidia.com/nvidia-tensorrt-download) 
- [CUDA 8.0](https://developer.nvidia.com/cuda-downloads)


## Install
### Requirements:
 1. CUDA 8.0 
 2. Ubuntu 16.04
 3. Download TensorRT2.1 .deb package
 
 ### Getting Started：
 ```bash
 $ sudo dpkg -i nv-gie-repo-ubuntu1604-cuda8.0-trt2.1-20170614_1-1_amd64.deb
 $ sudo apt-get update
 $ sudo apt-get install tensorrt-2.1.2
```
  Verify your installation:
  ```bash
  dpkg -l | grep TensorRT
  ```
    you should see:
    libnvinfer-dev 3.0.2-1+cuda8.0 amd64 TensorRT development libraries and headers
    libnvinfer3 3.0.2-1+cuda8.0 amd64 TensorRT runtime libraries   tensorrt-2.1.2 3.0.2-1+cuda8.0 amd64 Meta package of         TensorRT


  Run and Test TensorRT2.1
```
$ cd /usr/src/tensorrt/samples
$ sudo make
$ cd ../bin/
$ giexec --deploy=mnist.prototxt --model=mnist.caffemodel --output=prob
```
如果無提供“--model”，則權重將會隨機生成


## Introduce
NVIDIA TensorRT2.1是一个C++庫，在NVIDIA GPU上能够實現高性能的推理（inference ）過程。TensorRT優化網路的方式有：對張量和層進行合併，轉換權重。

編譯TensorRT 2.1 要求GCC >= 4.8

TensorRT 2.1 現在支持以下layer類型：

 - **Activation**: 激活層，The Activation layer implements per-element activation functions. Supported activation types are ReLU, TanH and Sigmoid
 - **Convolution**: 捲積層，The Convolution layer computes a 3D (channel, height, width) convolution, with or without bias.
 - **Concatenation**: 聯集層，The concatenation layer links together multiple tensors of the same height and width across the channel dimension
 - **Deconvolution**： 反捲積層，The Deconvolution layer implements a deconvolution, with or without bias.     
 - **ElementWise**: The ElementWise, also known as Eltwise, layer implements per-element operations. Supported operations are sum, product, and maximum
 - **Fully-connected**: 全連接層，The FullyConnected layer implements a matrix-vector product, with or without bias
 - **LRN**: The LRN layer implements cross-channel Local Response Normalization
 - **Plugin**: The Plugin Layer allows you to integrate layer implementations that TensorRT does not natively support
 - **Pooling**: 池化層，The Pooling layer implements pooling within a channel. Supported pooling types are maximum and average
 - **RNN**： 循環網路層，The RNN layer implements recurrent layers. Supported types are simple RNN, GRU,and LSTM.
 - **Scale**: The Scale layer implements a per-tensor, per channel or per-weight affine transformation and/or exponentiation by constant values
 - **SoftMax**: Softmax層，The SoftMax layer implements a cross-channel SoftMax.


雖然TensorRT獨立於任何框架，但該package確實包含一個名為NvCaffeParser的Caffe模型的解析器。 NvCaffeParser提供了一種導入網絡定義的簡單機制。 NvCaffeParser使用TensorRT的層來實現Caffe的Convolution,，ReLU，Sigmoid，TanH，Pooling，Power，BatchNorm，ElementWise（Eltwise），LRN，InnerProduct（在Caffe稱為FullyConnected層），SoftMax，Scale和Deconvolution層。而目前，NvCaffeParse不支持下面的Caffe層：

- Deconvolution groups
- Dilated convolutions
- PReLU
- Leaky ReLU
- Scale, other than per-channel scaling
- ElementWise (Eltwise) with more than two inputs

**Note：** NvCaffeParser不支持Caffe prototxt中的舊格式 


## Data Format
TensorRT2.1的輸入輸出張量均以NCHW形式儲存的32-bit張量。NCHW指張量的维度順序為batch维（N）-通道维（C）-高度（H）-寬度（W）





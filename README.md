
## Download TensorRT2.1
- [TensorRT2.1](https://developer.nvidia.com/nvidia-tensorrt-download) 



## 介绍
NVIDIA TensorRT2.1是一个C++庫，在NVIDIA GPU上能够實現高性能的推理（inference ）過程。TensorRT優化網路的方式有：對張量和層進行合併，轉換權重。

編譯TensorRT 2.1 要求GCC >= 4.8

TensorRT 2.1 現在支持以下layer類型：

 - **Activation**: 激活層，The Activation layer implements per-element activation functions. Supported activation types are ReLU, TanH and Sigmoid
 - **Convolution**:捲積層，The Convolution layer computes a 3D (channel, height, width) convolution, with or without bias.
 - **Concatenation**: 聯集層，The concatenation layer links together multiple tensors of the same height and width across the channel  dimension
 - **Deconvolution**： 反捲積層，The Deconvolution layer implements a deconvolution, with or without bias.     
 - **ElementWise**: The ElementWise, also known as Eltwise, layer implements per-element operations. Supported operations are sum,    product, and maximum
 - **Fully-connected**：全連接層，The FullyConnected layer implements a matrix-vector product, with or without bias
 - **LRN**:The LRN layer implements cross-channel Local Response Normalization
 - **Plugin**:The Plugin Layer allows you to integrate layer implementations that TensorRT does not natively support
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
引號。

## 快速开始指南

TensorRT原名GIE。GIE又名TensorRT 1.0，TensorRT 2.0正式改名。
TensorRT 2.0非常大的改动点是支持INT8类型（TensorRT 1.0支持FP16）。
使用TensorRT 2.0的硬件要求：Tesla P4, Tesla P40, GeForce TitanX Pascal, GeForce GTX 1080, DRIVE PX 2 dGPU
软件要求：CUDA 8.0
### Ubuntu 下安装方式
安装命令：

 1. CUDA 8.0 
 2. Ubuntu 16.04
 3. Download TensorRT2.1 .deb package
 4. Install TensrRT2.1 .deb command line：
 ``
    sudo dpkg -i nv-gie-repo-ubuntu1604-cuda8.0-trt2.1-20170614_1-1_amd64.deb
    sudo apt-get update
    sudo apt-get install tensorrt-2.1.2
``
 4. Verify your installation:
  ```bash
  dpkg -l | grep TensorRT
  ```
    you should see:
    libnvinfer-dev 3.0.2-1+cuda8.0 amd64 TensorRT development libraries and headers
    libnvinfer3 3.0.2-1+cuda8.0 amd64 TensorRT runtime libraries   tensorrt-2.1.2 3.0.2-1+cuda8.0 amd64 Meta package of         TensorRT


 5. Run and Test TensorRT2.1
```
$ cd /usr/src/tensorrt/samples
$ sudo make
$  cd ../bin/
$ giexec --deploy=mnist.prototxt --model=mnist.caffemodel --output=prob
```
如果無提供“--model”，则全重將會隨機生成

该样例没有展示任何前述未曾包含的TensorRT特性

## 在多GPU上使用TensorRT
每个`ICudaEngine`对象在通过builder或反序列化而实例化时均被builder限制于一个指定的GPU内。要进行GPU的选择，需要在进行反序列化或调用builder之前调用`cudaSetDeviec()`。每个`IExecutionContext`都被限制在产生它的引擎所在的GPU内，当调用`execute()`或`enqueue()`时，请在必要时调用`cudaSetDevice()`以保证线程与正确的设备绑定。

## 数据格式
TensorRT的输入输出张量均为以NCHW形式存储的32-bit张量。NCHW指张量的维度顺序为batch维（N）-通道维（C）-高度（H）-宽度（W）

对权重而言：

- 卷积核存储为KCRS形式，其中K轴为卷积核数目的维度，即卷积层输出通道维。C轴为是输入张量的通道维。R和S分别是卷积核的高和宽
- 全连接层按照行主序形式存储  <font color="red">这里是错的！！全连接层中weights的存储方式是col-major，详见[Bugs](https://github.com/LitLeo/TensorRT_Tutorial/blob/master/Bug.md)</font>
- 反卷积层按照CKRS形式存储，各维含义同上

## FAQ
**Q：如何在TensorRT中使用自定义层？**
A：当前版本的TensorRT不支持自定义层。要想在TensorRT中使用自定义层，可以创建两个TensorRT工作流，自定义层夹在中间执行。比如：

``` c++
IExecutionContext *contextA = engineA->createExecutionContext();
IExecutionContext *contextB = engineB->createExecutionContext();

<...>

contextA.enqueue(batchSize, buffersA, stream, nullptr);
myLayer(outputFromA, inputToB, stream);
contextB.enqueue(batchSize, buffersB, stream, nullptr);
```

**Q：如何构造对若干不同可能的batch size优化了的引擎？**
A：尽管TensorRT允许在给定的一个batch size下优化模型，并在运行时送入任何小于该batch size的数据，但模型在更小size的数据上的性能可能没有被很好的优化。为了面对不同batch大小优化模型，你应该对每种batch size都运行一下builder和序列化。未来的TensorRT可能能基于单一引擎对多种batch size进行优化，并允许在当不同batch size下层使用相同的权重形式时，共享层的权重。

**Q：如何选择最佳的工作空间大小**:
A: 一些TensorRT算法需要GPU上额外的工作空间。方法`IBuilder::setMaxWorkspaceSize()`控制了能够分配的工作空间的最大值，并阻止builder考察那些需要更多空间的算法。在运行时，当创造一个`IExecutionContext`时，空间将被自动分配。分配的空间将不多于所需求的空间，即使在`IBuilder::setMaxWorspaceSize()`中设置的空间还有很多。应用程序因此应该允许TensorRT builder使用尽可能多的空间。在运行时，TensorRT分配的空间不会超过此数，通常而言要少得多。


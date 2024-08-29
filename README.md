## ming-NN

一个C++简易神经网络头文件库
使用C++17标准（编译时请注意您的编译器版本)


使用仿照Keras，将各层叠加在一起，已经实现的层包括：

```cpp
mingNN::Layers::Input
mingNN::Layers::Dense
mingNN::Layers::ReLU
mingNN::Layers::SoftMax
mingNN::Layers::DropOut
```

### 运行示例

直接在目录下进行简单编译即可

使用命令：

g++ -std\=c++17 test.cpp

## 特点

简易实现Cmat矩阵运算，并使用Mr.Barak Shoshany实现的BS\_thread\_pool线程池库(<https://baraksh.com/>)对矩阵运输进行并行优化

### 目录结构

│  IrisDataset.hpp
│  README.md
│  test.cpp
│
├─.vscode
│      settings.json
│
└─include
    │  Layer.hpp
    │  Loss.hpp
    │  mingNN.hpp
    │  Optimizer.hpp
    │  SequentialModel.hpp
    │  utils.hpp
    │
    ├─Layers
    │      Dense.hpp
    │      DropOut.hpp
    │      Input.hpp
    │      ReLU.hpp
    │      SoftMax.hpp
    │
    │
    ├─lib
    │  ├─Cmat
    │  │      Cmat.hpp
    │  │      common.hpp
    │  │      Cmat.hpp
    │  │      common.hpp
    │  │      matrix.hpp
    │  │      common.hpp
    │  │      matrix.hpp
    │  │
    │  │      matrix.hpp
    │  │
    │  └─thread-pool
    │  │
    │  └─thread-pool
    │          BS_thread_pool.hpp
    │          BS_thread_pool_light.hpp
    │  └─thread-pool
    │          BS_thread_pool.hpp
    │          BS_thread_pool_light.hpp
    │          BS_thread_pool.hpp
    │          BS_thread_pool_light.hpp
    │
    │          BS_thread_pool_light.hpp
    │
    │
    ├─Loss
    │      CrossEntropy.hpp
    │      MeanSqure.hpp
    │
    └─Optimizers
            SGD.hpp

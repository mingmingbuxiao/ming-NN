## ming-NN

神经网络头文件库

使用C++17标准，编译时请注意您的编译器版本



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

> &#x20;


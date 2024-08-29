#include <iostream>                        
#include <vector>                          
#include "include/mingNN.hpp"               
#include "IrisDataset.hpp"                 

int main () {                              
    mingNN::SequentialModel model;          // 创建一个顺序模型对象
    model.addLayer<mingNN::Layers::Input>({4});  // 添加输入层
    model.addLayer<mingNN::Layers::ReLU>({});    // 添加ReLU激活函数层
    model.addLayer<mingNN::Layers::Dense>({128}); // 添加全连接层
    model.addLayer<mingNN::Layers::DropOut>({0,0.5}); // 添加 dropout 层，第二个参数丢弃概率
    model.addLayer<mingNN::Layers::ReLU>({});    // 添加ReLU激活函数层
    model.addLayer<mingNN::Layers::Dense>({3});  // 添加全连接层
    model.addLayer<mingNN::Layers::SoftMax>({}); // 添加SoftMax激活函数层

    model.compile<mingNN::Optimizer::SGD, mingNN::Loss::CrossEntropy>({0.001}); // 编译模型，使用随机梯度下降（SGD）优化器和交叉熵损失函数，学习率为0.001

    mingNN::XorShift rnd;                  // 创建一个随机数生成器对象
    Cmat::Matrix x, y;                    // 定义输入矩阵x和标签矩阵y

    for (auto& data : IRIS_DATASET)       // 遍历鸢尾花数据集
    {
        x.push_back(data.data);           // 将数据样本添加到输入矩阵x
        std::vector<double> t(3);         // 创建一个长度为3的向量t，用于独热编码
        t[data.kind] = 1;                 // 设置对应类别的索引为1
        y.push_back(t);                   // 将独热编码向量t添加到标签矩阵y
    }

    for (int step = 0; step < 1000; ++step) // 训练模型
    {
        Cmat::Matrix bx, by;              // 定义批次输入矩阵bx和批次标签矩阵by

        for (int i = 0; i < 50; ++i)      // 每个批次随机选择50个样本
        {
            int idx = rnd() * IRIS_DATASET.size(); // 随机生成样本索引
            bx.push_back(x[idx]);         // 将随机选择的输入样本添加到批次输入矩阵bx
            by.push_back(y[idx]);         // 将对应的标签添加到批次标签矩阵by
        }

        auto t = model.forward(bx);       // 执行前向传播，计算输出
        auto loss = model.eval(t, by);    // 计算损失值
        model.backward();                 // 执行反向传播，更新模型参数

        if ((step + 1) % 100 == 0) {      // 每训练100步输出一次信息
            int accepted = 0;             // 初始化正确预测计数
            t = model.forward(x);         // 对整个数据集执行前向传播

            for (int i = 0; i < t.m_height; ++i) // 遍历输出矩阵
            {
                int a = std::max_element(t[i].begin(), t[i].end()) - t[i].begin(); // 找到输出最大值的索引
                int b = std::max_element(y[i].begin(), y[i].end()) - y[i].begin(); // 找到标签最大值的索引
                accepted += (a == b);     // 如果预测正确，增加计数
            }

            std::cout << "step:" << step + 1 << " loss:" << loss << " accuracy:" << (double)accepted / t.m_height * 100 << "%" << std::endl; // 输出当前训练步的损失和准确率
        }
    }

    return 0; // 返回0，结束程序
}

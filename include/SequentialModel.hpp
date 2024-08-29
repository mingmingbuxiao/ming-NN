#pragma once
#include <vector>
#include <memory>
#include <assert.h>
#include "lib/Cmat/Cmat.hpp"
#include "Layer.hpp"
#include "utils.hpp"
#include "Optimizer.hpp"
#include "Loss.hpp"

namespace mingNN
{

// 定义一个结构体 SequentialModel，用于构建顺序神经网络模型
struct SequentialModel
{
    // 成员变量
    std::vector<std::shared_ptr<Layers::Layer>> m_layers; // 存储网络中的各个层的指针
    std::shared_ptr<Loss::Loss> m_loss; // 存储损失函数的指针
    int m_input = 0, m_output = 0; // 输入和输出的维度
    bool compiled = false; // 标记模型是否已经编译

    // 默认构造函数
    SequentialModel() {}

    // 析构函数
    ~SequentialModel() {}

    // 模板函数，添加新的一层到模型中
    template<class T>
    void addLayer(Layers::InitData init)
    {
        // 使用make_shared创建新的层，并添加到m_layers向量中
        // 这里假设T是继承自Layers::Layer的某个层类
        m_layers.push_back(std::make_shared<T>(m_input, m_output, init));
    }

    // 模板函数，编译模型
    template<class Opt, class Loss>
    void compile(Optimizer::InitData init)
    {
        // 创建损失函数的共享指针
        m_loss = std::make_shared<Loss>();
        compiled = true; // 标记模型已编译

        // 为模型中的每一层编译优化器
        for (auto& layer : m_layers)
        {
            layer->compile<Opt>(init);
        }
    }

    // 前向传播函数
    Cmat::Matrix forward(Cmat::Matrix x)
    {
        // 依次通过每一层进行前向传播
        for (auto& layer : m_layers)
        {
            x = layer->forward(x);
        }
        return x;
    }

    // 评估函数，计算损失
    // y: 预测输出, t: 真实标签
    double eval(const Cmat::Matrix& y, const Cmat::Matrix& t)
    {
        // 确保模型已经编译
        assert(compiled);

        // 使用损失函数计算损失值
        return m_loss->forward(y, t);
    }

    // 反向传播函数
    Cmat::Matrix backward()
    {
        // 确保模型已经编译
        assert(compiled);

        // 计算损失的梯度
        Cmat::Matrix x = m_loss->backward();

        // 依次通过每一层进行反向传播（从最后一层到第一层）
        for (int i = m_layers.size() - 1; i >= 0; --i)
        {
            x = m_layers[i]->backward(x);
        }
        return x;
    }
};

}

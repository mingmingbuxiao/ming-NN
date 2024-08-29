#pragma once
#include "../Layer.hpp"
#include "../utils.hpp"
#include "../Optimizer.hpp"

namespace mingNN
{

namespace Layers
{

class Dense: public Layer
{
private:
    // 神经元权重矩阵和偏置矩阵，使用智能指针管理内存
    std::shared_ptr<Cmat::Matrix> m_neuron, m_bias;
    // 优化器对象的智能指针，用于更新神经元权重和偏置
    std::shared_ptr<Optimizer::Optimizer>m_opt_neuron, m_opt_bias;
    // 上一次输入的存储矩阵
    Cmat::Matrix m_last;
    // 单元数量
    int m_unit;
    // CPU 核心数量，默认设置为可用的硬件线程数
    size_t n_cpu = std::thread::hardware_concurrency();
    // 输入和输出单元的数量
    int m_input, m_output;
    // 随机数生成器，用于初始化权重和偏置
    XorShift rnd;
    // 线程池对象，用于并行化计算
    BS::thread_pool pool;

public:
    // 构造函数，初始化层的权重和偏置，并使用随机数生成器进行初始化
    Dense([[maybe_unused]]int& input, int& output, InitData init): m_unit(init.unit), pool(n_cpu)
    {
        // 初始化神经元权重矩阵和偏置矩阵
        m_neuron = std::make_shared<Cmat::Matrix>(output, init.unit);
        m_bias = std::make_shared<Cmat::Matrix>(1, init.unit);

        // 使用He初始化法计算标准差（σ），用于权重初始化
        double sigma = sqrt(2.0 / m_unit);

        // 初始化神经元权重和偏置，使用正态分布的随机数
        for (int j = 0; j < init.unit; ++j)
        {
            for (int i = 0; i < output; ++i)
            {
                (*m_neuron)[i][j] = rnd.normal() * sigma;
            }
            (*m_bias)[0][j] = rnd.normal() * sigma;
        }

        // 设置输入输出维度
        m_input = output;
        m_output = init.unit;
        output = init.unit;
    }

    // 编译层的函数，初始化优化器
    void compile(Optimizer::InitData init) override
    {
        // 根据初始化数据为神经元和偏置创建优化器
        m_opt_neuron = getOptimizer(init, m_neuron);
        m_opt_bias = getOptimizer(init, m_bias);
    }

    // 返回层的名称
    std::string LayerName() override 
    {
        return "Dense";
    }

    // 前向传播函数，输入矩阵x，输出矩阵
    Cmat::Matrix forward(Cmat::Matrix x) override
    {
        // 保存输入矩阵
        m_last = x;

        // 计算输出矩阵：输入矩阵与权重矩阵相乘
        x = x.dot(*m_neuron);

        // 多线程计算加上偏置后的输出矩阵
        for (int i = 0; i < x.m_height; ++i)
        {
            pool.push_task([&](int i){
                for (int j = 0; j < m_output; ++j)
                {
                    x[i][j] += (*m_bias)[0][j];
                }
            }, i);
        }

        // 等待所有任务完成
        pool.wait_for_tasks();

        return x;
    }

    // 反向传播函数，计算梯度并更新权重和偏置
    Cmat::Matrix backward(Cmat::Matrix x) override
    {
        // 计算输入的梯度
        Cmat::Matrix dx = x.dot(m_neuron->transpose());
        // 计算权重的梯度
        Cmat::Matrix dw = m_last.transpose().dot(x);
        // 初始化偏置的梯度矩阵
        Cmat::Matrix db(1, x[0].size());

        // 多线程计算偏置的梯度
        for (int i = 0; i < x.m_height; ++i)
        {
            pool.push_task([&](int i){
                for (int j = 0; j < x.m_width; ++j)
                {
                    db[0][j] += x[i][j];
                }
            }, i);
        }

        // 等待所有任务完成
        pool.wait_for_tasks();

        // 更新权重和偏置
        m_opt_neuron->step(dw);
        m_opt_bias->step(db);

        return dx;
    }
};

}

}

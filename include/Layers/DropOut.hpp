#pragma once

#include "../Layer.hpp"
#include <random>

namespace mingNN
{

namespace Layers
{

class DropOut : public Layer
{
private:
    Cmat::Matrix mask; // 掩码矩阵，用于决定哪些神经元被丢弃
    double dropout_ratio; // dropout比率
    std::default_random_engine generator; // 随机数生成器
    std::bernoulli_distribution distribution; // 伯努利分布，用于生成布尔型随机数

public:
    // 构造函数，初始化dropout比率
    DropOut([[maybe_unused]] int &input, [[maybe_unused]] int &output, InitData init)
    {
        // 使用断言确保 dropout_ratio 在 0 到 1 之间
        assert(init.dropout_ratio > 0.0 && init.dropout_ratio < 1.0 && "Dropout ratio must be between 0 and 1");

        // 如果比率在范围内，则初始化 dropout_ratio
        dropout_ratio = init.dropout_ratio;

        distribution = std::bernoulli_distribution(dropout_ratio);
    }


    // 返回层的名称
    std::string LayerName() override { return "DropOut"; }

    void compile([[maybe_unused]]Optimizer::InitData init) override {};

    // 前向传播
    Cmat::Matrix forward(Cmat::Matrix x) override
    {
        // 生成与输入矩阵相同尺寸的掩码矩阵
        mask = Cmat::Matrix(x.m_height, x.m_width);

        // int zero_count = 0; // 统计 0 的数量
        // 随机生成掩码
        for (int i = 0; i < mask.m_height; ++i)
        {
            for (int j = 0; j < mask.m_width; ++j)
            {
                // 使用伯努利分布生成布尔值来决定掩码元素是否为1
                mask[i][j] = distribution(generator) ? 1 : 0;
                // if (mask[i][j] == 0) {
                //     zero_count++; // 统计 0 的数量
                // }
            }
        }
        // // 计算 0 的占比
        // double zero_ratio = static_cast<double>(zero_count) / (mask.m_height * mask.m_width);
        // // 输出 0 的占比
        // std::cout << "Zero ratio in mask: " << zero_ratio << std::endl;


        // 将输入矩阵与掩码矩阵逐元素相乘
        x *= mask;

        // 返回 dropout 后的矩阵
        return x;
    }

    // 反向传播
    Cmat::Matrix backward(Cmat::Matrix x) override
    {
        // 仅保留前向传播时保留的神经元的梯度
        x *= mask;

        // 返回经过掩码缩放的梯度
        return x;
    }
};

}


}

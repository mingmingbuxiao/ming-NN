#pragma once
#include "../Loss.hpp"

namespace mingNN
{

namespace Loss
{

// 定义一个均方误差（Mean Squared Error, MSE）损失类，继承自基类 Loss
class MeanSqure: public Loss
{
private:
    Cmat::Matrix m_back; // 用于存储反向传播时所需的梯度

public:
    // 前向传播函数，计算均方误差损失
    // x: 预测值 (predict)，t: 真实标签 (answer)
    double forward(const Cmat::Matrix& x, const Cmat::Matrix& t) override
    {
        auto err = (x - t); // 计算预测值和真实标签的差值，得到误差矩阵

        m_back = err; // 将误差矩阵保存到 m_back 中，用于反向传播计算梯度

        err *= err * 0.5; // 逐元素平方误差并乘以 0.5，计算 MSE 的每项损失

        double res = err.sum() / err.m_height; // 计算总损失，取所有样本的平均损失值

        return res; // 返回均方误差损失
    }

    // 反向传播函数，返回损失函数的梯度
    Cmat::Matrix backward() override
    {
        return m_back; // 返回前向传播中保存的误差矩阵作为梯度
    }
};

}

}

#pragma once
#include <math.h>
#include <cmath>
#include "../Loss.hpp"
#include "../utils.hpp"

namespace mingNN
{

namespace Loss
{

// 定义一个交叉熵损失类，继承自基类 Loss
class CrossEntropy: public Loss
{
private:
    Cmat::Matrix m_back; // 存储反向传播所需的梯度
    BS::thread_pool pool = BS::thread_pool(std::thread::hardware_concurrency()); // 线程池，用于并行计算

public:
    // 前向传播函数，计算交叉熵损失
    // x: 预测值 (predict)，t: 真实标签 (answer)
    double forward(const Cmat::Matrix& x, const Cmat::Matrix& t) override
    {
        auto err = x; // 将预测值拷贝到错误矩阵中
        
        // 并行计算对数
        for (int i = 0; i < err.m_height; ++i)
        {
            pool.push_task([&](int i){
                for (int j = 0; j < err.m_width; ++j)
                {
                    // 防止log(0)的情况，增加小量0.0001
                    if (isnan(log(err[i][j] + 0.0001)))
                    {
                        std::cout << err[i][j] << std::endl; // 输出发生问题的值
                        assert(false); // 断言失败，中止程序
                    }
                    err[i][j] = log(err[i][j] + 0.0001); // 计算对数
                }
            }, i);
        }
        pool.wait_for_tasks(); // 等待所有线程任务完成

        // 计算交叉熵损失
        err *= t; // 逐元素相乘，计算交叉熵的部分和
        m_back = x - t; // 计算反向传播所需的梯度，保存为 m_back

        return err.sum() / err.m_height * -1; // 计算平均损失，并取负值
    }

    // 反向传播函数，返回损失函数的梯度
    Cmat::Matrix backward() override
    {
        return m_back; // 返回之前在前向传播时计算的梯度
    }
};

}

}

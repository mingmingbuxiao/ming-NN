#pragma once
#include <string>
#include <memory>
#include "lib/Cmat/Cmat.hpp"

// 防止重复包含头文件
// #pragma once 是一种预处理器指令，用于确保头文件只被编译器编译一次，
// 避免重复包含可能导致的重定义错误。

namespace mingNN
{

namespace Optimizer
{

// 定义一个结构体 InitData，用于优化器的初始化参数
struct InitData
{
    double lr = 0; // 学习率 (Learning Rate)，用于控制优化器步长，默认为0。
};

// 定义一个抽象基类 Optimizer，用于实现不同类型的优化器
class Optimizer
{
private:
    std::shared_ptr<Cmat::Matrix> m_target; 
    // m_target 是一个 shared_ptr 指针，指向一个 Cmat::Matrix 对象，
    // 该对象代表了优化器将要更新的参数矩阵。

public:
    // 构造函数，初始化优化器并绑定一个要优化的目标矩阵
    Optimizer(std::shared_ptr<Cmat::Matrix> target): m_target(target) {}

    // 纯虚函数，返回优化器的名称，必须在派生类中实现
    virtual std::string OptimizerName() = 0;

    // 纯虚函数，执行一步优化，更新目标矩阵，必须在派生类中实现
    virtual void step(const Cmat::Matrix& grad) = 0;

    // 运算符重载，使得对象可以像函数一样被调用，内部调用 step() 方法
    void operator()(const Cmat::Matrix& grad)
    {
        step(grad);
    }

protected:
    // 保护成员函数，返回优化器目标矩阵的指针
    // 只能在类内部或派生类中调用，用于访问 m_target
    std::shared_ptr<Cmat::Matrix> getTarget()
    {
        return m_target;
    }
};

}

}

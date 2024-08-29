#pragma once
#include <memory>
#include <vector>
#include <assert.h>
#include <thread>
#include <iostream>
# include "../thread-pool/BS_thread_pool.hpp"

namespace Cmat
{

struct Matrix 
{
    int m_width, m_height; // 矩阵宽度和高度
    std::vector<std::vector<double>> m_data; // 存储矩阵数据的二维向量
    size_t n_cpu; // 并发线程数
    std::shared_ptr<BS::thread_pool> pool; // 线程池指针

    Matrix():
        m_width(0), 
        m_height(0),
        n_cpu(std::thread::hardware_concurrency()),// 获取硬件并发线程
        pool(std::make_shared<BS::thread_pool>(n_cpu))// 创建线程池
    {}
    ~Matrix() {}

    Matrix(std::vector<double>v):// 从一维向量构造矩阵
        n_cpu(std::thread::hardware_concurrency()),
        pool(std::make_shared<BS::thread_pool>(n_cpu))
    {
        m_data = {v};
        m_width = v.size();
        m_height = 1;
    }

    Matrix(std::vector<std::vector<double>>v):// 从二维向量构造矩阵
        n_cpu(std::thread::hardware_concurrency()),
        pool(std::make_shared<BS::thread_pool>(n_cpu))
    {
        m_data = v;
        m_width = v.size()?v[0].size():0;
        m_height = v.size();
    }

    Matrix(int height, int width):// 从高度和宽度构造零矩阵
        n_cpu(std::thread::hardware_concurrency()),
        pool(std::make_shared<BS::thread_pool>(n_cpu))
    {
        *this = Matrix::zeros(height, width);// 调用zeros静态函数创建零矩阵
    }

    Matrix zeros(size_t height, size_t width) const {// 创建指定大小的零矩阵
        return Matrix{ std::vector<std::vector<double>>(height, std::vector<double>(width, 0)) };
    }

    void push_back(std::vector<double>row)// 在矩阵末尾添加一行
    {
        assert((int)row.size() == m_width or m_height==0);// 检查行长度是否与矩阵宽度一致

        m_data.push_back(row);

        if (m_height == 0) m_width = row.size();// 更新矩阵宽度
        m_height++;
    }

    void print()// 打印矩阵
    {
        for (int i=0; i<m_height; ++i)
        {
            for (int j=0; j<m_width; ++j)
            {
                std::cout << m_data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    Matrix operator+(const Matrix m) const { return Matrix(*this) += m; }
    Matrix operator-(const Matrix m) const { return Matrix(*this) -= m; }
    Matrix operator*(const Matrix m) const { return Matrix(*this) *= m; }
    Matrix operator*(const double a) const { return Matrix(*this) *= a; }
    Matrix operator/(const Matrix m) const { return Matrix(*this) /= m; }

    Matrix operator+=(const Matrix m)// 矩阵加法赋值运算符
    {
        assert(m_width!=m.m_width and m_height!=m.m_height);// 确保两个矩阵的尺寸相同，断言检查，不符合就终止程序

        for (int i = 0; i < m_height; ++i)// 分行计算，对于每一行，调用 pool->push_task 将加法操作作为一个任务添加到线程池中
        {
            pool->push_task([&](int i){// 使用线程池并行计算，匿名函数
                for (int j = 0; j < m_width; ++j)
                {
                    m_data[i][j] = m_data[i][j] + m.m_data[i][j];
                }
            }, i);
        }
        pool->wait_for_tasks();// 等待所有任务完成

        return *this;
    }

    Matrix operator-=(const Matrix m)// 矩阵减法赋值运算符
    {
        for (int i = 0; i < m_height; ++i)
        {
            pool->push_task([&](int i){
                for (int j = 0; j < m_width; ++j)
                {
                    m_data[i][j] = m_data[i][j] - m.m_data[i][j];
                }
            }, i);
        }
        pool->wait_for_tasks();

        return *this;
    }

    Matrix operator*=(const double a)// 矩阵数乘赋值运算符
    {
        for (int i = 0; i < m_height; ++i)
        {
            pool->push_task([&](int i){
                for (int j = 0; j < m_width; ++j)
                {
                    m_data[i][j] = m_data[i][j] * a;
                }
            }, i);
        }
        pool->wait_for_tasks();

        return *this;
    }

    Matrix operator*=(const Matrix m)// 矩阵乘法赋值运算符
    {
        for (int i = 0; i < m_height; ++i)
        {
            pool->push_task([&](int i){
                for (int j = 0; j < m_width; ++j)
                {
                    m_data[i][j] = m_data[i][j] * m.m_data[i][j];
                }
            }, i);
        }
        pool->wait_for_tasks();

        return *this;
    }

    Matrix operator/=(const Matrix m)// 矩阵除法赋值运算符
    {
        for (int i = 0; i < m_height; ++i)
        {
            pool->push_task([&](int i){
                for (int j = 0; j < m_width; ++j)
                {
                    m_data[i][j] = m_data[i][j] / m.m_data[i][j];
                }
            }, i);
        }
        pool->wait_for_tasks();

        return *this;
    }

    std::vector<double>& operator[](int i)// 重载下标运算符
    {
        return m_data[i];
    }

    Matrix dot(const Matrix m)// 矩阵点乘
    {
        assert(m_width == m.m_height);

        Matrix mat = Matrix::zeros(m_height, m.m_width);

        for (int idx = 0; idx < m_height; ++idx)
        {
            pool->push_task([&](int idx){
                for (int i=0; i<m_width; ++i)
                {
                    for (int j=0; j<m.m_width; ++j)
                    {
                        mat.m_data[idx][j] += m_data[idx][i] * m.m_data[i][j];
                    }
                }
            }, idx);
        }

        pool->wait_for_tasks();

        return mat;
    }

    Matrix transpose() const // 矩阵转置
    {
        Matrix mat{ m_width, m_height };

        for (int i = 0; i < m_height; ++i)
        {
            pool->push_task([&](int i){
                for (int j = 0; j < m_width; ++j)
                {
                    mat.m_data[j][i] = m_data[i][j];
                }
            }, i);
        }
        pool->wait_for_tasks();

        return mat;
    }

    double sum() const// 矩阵元素之和
    {
        double res = 0;

        for (int i = 0; i < m_height; ++i)
        {
            pool->push_task([&](int i){
                for (int j = 0; j < m_width; ++j)
                {
                    res += m_data[i][j];
                }
            }, i);
        }

        pool->wait_for_tasks();

        return res;
    }
};

}
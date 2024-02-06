//
// Created by mzh on 2024/1/29.
//

#include "minfer.h"
#include "gtest/gtest.h"

#include <iostream>

using namespace std;

using namespace minfer;

//A是一个父类 , 析构函数不是虚函数
class A
{
public:
    A()
    {
        cout << " A constructor" << endl;
        fun();
    }

    void fun()
    {
        cout << " A fun" << endl;
    }
    ~A()
    {
        cout << " A destructor" << endl;
    }
};

//B是A的子类
class B : public A
{
public:
    B()
    {
        cout << " B constructor" << endl;
        fun();
    }

    void fun()
    {
        cout << " B fun" << endl;
    }

    ~B()
    {
        cout << " B destructor" << endl;
    }
};

//C是一个父类 , 析构函数是虚函数
class C
{
public:
    C()
    {
        cout << " C constructor" << endl;
        fun();
    }

    virtual void fun()
    {
        cout << " C fun" << endl;
    }

    virtual ~C()
    {
        cout << " C destructor" << endl;
    }
};

//D是C的子类
class D : public C
{
public:

    D()
    {
        cout << " D constructor" << endl;
        fun();
    }

    virtual void fun()
    {
        cout << " D fun" << endl;
    }

    ~D()
    {
        cout << " D destructor" << endl;
    }
};

TEST(Tensor, create_0)
{
//    Tensor t = Tensor({1}, Tensor::DataType(1));
//    Tensor t2 = Tensor(); // 这个走的是Tensor(tensor)这个构造函数

    A *a = new B();
//    a->fun();
    delete a;
    cout << "-----------------------------"<<endl;
    B * b = new B();
//    b->fun();
    delete b;
    cout << "-----------------------------"<<endl;
    C *c = new D();
//    c->fun();
    delete c;
}

#define CV_SINGLETON_LAZY_INIT_(TYPE, INITIALIZER, RET_VALUE) \
    static TYPE* const instance = INITIALIZER;
//    return RET_VALUE;

#define CV_SINGLETON_LAZY_INIT(TYPE, INITIALIZER) CV_SINGLETON_LAZY_INIT_(TYPE, INITIALIZER, instance)

TEST(Tensor, create_1)
{
    std::cout<<"Tensor 1"<<std::endl;
    CV_SINGLETON_LAZY_INIT(A, new A());
//    CV_SINGLETON_LAZY_INIT(A, new A());
//    CV_SINGLETON_LAZY_INIT(A, new A());
//    return true;
}

TEST(Tensor, add)
{
    std::cout<<"Tensor 2"<<std::endl;
    CV_SINGLETON_LAZY_INIT(A, new A());
}

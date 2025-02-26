#include <iostream>
#include <string>
//1:指针什么时候用？
//2：索引的实质其实是别人给你写好的一种查数据的指针代码，你只需要调用一下就可以查到你想要的数据。
//3：std::是一个仓库
//4：类其实就是一个事物，它有自己的属性和方法，可以用来描述事物的特征和行为，特征就是数据，行为就是函数。
//5：构造函数的作用是用来初始化类的实例的，它可以有多个，但是只有一个构造函数可以被调用，
//构造函数重载作用·是为了让类可以根据不同的参数来实例化，使得类可以处理不同的数据类型。
class Person {
public:
  std::string name;
  int age;

public:
  Person(std::string name, int age) {
    this->name = name;
    this->age = age;
  }
  
  Person() {
    this->name = "ccy";
    this->age = 18;
  }
  Person(int age) {
    this->name = "ccy";
    this->age = age;
  }
  /*哒咩
  Person(int m_age) { 
    this->name = "ccy";
    this->age = m_age;};
 */
  Person(double m_age) {
    this->name = "ccy";
    this->age = m_age;
  }
  void sayHello() {
    std::cout << "Hello, my name is " << name << " and I am " << age
              << " years old." << std::endl;
  }
};
class Phone{
  public:
    std::string brand;
    std::string color;
    int price;
    virtual void call()=0;
};
class ApplePhone:public Phone{
  public:
    void call(){
      std::cout<<"calling from ApplePhone"<<std::endl;
    }
};
class xiaomiPhone:public Phone{
  public:
    void call(){
      std::cout<<"calling from xiaomiPhone"<<std::endl;
    }
};
void call(Phone* p){
  p->call();
}
int main() {
  
  Person p1("John", 52);
  Person p2;
  Person p3(25.1);
  p2.sayHello();
  p1.sayHello();
  p3.sayHello();
  ApplePhone apple;
  xiaomiPhone xiaomi;
  call(&apple);
  call(&xiaomi);
  return 0;
}
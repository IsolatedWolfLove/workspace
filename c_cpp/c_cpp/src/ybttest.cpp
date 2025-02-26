#include <iostream>
#include <cmath>
using namespace std;
class shape{
    public:
    virtual double area() const=0;
    void printarea()
    {cout<<"the area is:"<<area()<<endl;}
};
class circle:public shape{
    private:
    double r;
    public:
    void setr(double r){this->r=r;}
    double getr()const{return r;}
    double area() const
    {double pi=3.14;
     return pi*r*r;}   
};
class rectangle:public shape{
    private:
    double l;double h;
    public:
    void setl(double l){this->l=l;}
    double getl(){return l;}
    void seth(double h){this->h=h;}
    double geth(){return h;}
    double area() const {return h*l;}
};
class triangle:public shape{
    private:
    double a;double b;double c;
    public:
    void seta(double a){this->a=a;}
    double geta()const{return a;}
    void setb(double b){this->b=b;}
    double getb()const{return b;}
    void setc(double c){this->c=c;}
    double getc()const{return c;}
    double area() const 
    {double l=(a+b+c)/2;
    return sqrt(l*(l-a)*(l-b)*(l-c));}
};
int main()
{circle circle;
 circle.setr(6);
 circle.area();
 circle.printarea();
 rectangle rectangle;
 rectangle.setl(3);
 rectangle.seth(4);
 rectangle.area();
 rectangle.printarea();
 triangle triangle;
 triangle.seta(3);
 triangle.setb(4);
 triangle.setc(5);
 triangle.area();
 triangle.printarea(); 
return 0;}
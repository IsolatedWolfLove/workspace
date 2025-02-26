#include <iostream>
#include <memory>
#include <ostream>
#include <utility>
class plural_num {
  friend std::ostream &operator<<(std::ostream &cout, plural_num &num);

public:
  plural_num(double num1, double num2) : num_pair(num1, num2){};

  plural_num operator+(plural_num &other) {
    this->num_pair.first += other.num_pair.first;
    this->num_pair.second += other.num_pair.second;
    return *this;
  }
  plural_num &operator*(int num) {
    this->num_pair.first *= num;
    this->num_pair.second *= num;
    return *this;
  }
  plural_num &operator*(plural_num &other) {
    // plural_num* temp=new plural_num(0,0);
    // std::shared_ptr<plural_num> temp(new plural_num(0,0));
    double temp1 = this->num_pair.first * other.num_pair.first -
                   this->num_pair.second * other.num_pair.second;
    double temp2 = this->num_pair.first * other.num_pair.second +
                   this->num_pair.second * other.num_pair.first;

    this->num_pair.first = temp1;
    this->num_pair.second = temp2;

    return *this;
  }

private:
  std::pair<double, double> num_pair;
};
std::ostream &operator<<(std::ostream &cout, plural_num &num) {
  std::cout << num.num_pair.first << "+" << num.num_pair.second << " i\n";
  return cout;
}
class three_d {
  friend three_d operator++(three_d &t);
  friend std::ostream& operator<<(std::ostream &cout, three_d& t);

public:
  three_d(double x, double y, double z) : x_(x), y_(y), z_(z){};
  three_d& operator++() {
        ++x_;
        ++y_;
        ++z_;
        return *this;
    }
  three_d operator++(int) {
        three_d temp = *this; // 保存当前状态
        ++x_; ++y_; ++z_; // 修改当前对象
        return temp; // 返回保存的状态
    }

private:
  double x_;
  double y_;
  double z_;
};
std::ostream& operator<<(std::ostream &cout, three_d& t) {
  std::cout << "( " << t.x_ << "," << t.y_ << "," << t.z_ << ")" << std::endl;
  return cout;
}
 three_d operator++(three_d &t){
    three_d temp = t; // 保存当前状态
    ++t.x_; ++t.y_; ++t.z_; // 修改当前对象
    return temp; // 返回保存的状态  
 };
void test01() {
  plural_num num1(2, 3);
  plural_num num2(4, 5);
  plural_num num3 = num1 + num2;
  std::cout << num3;
  plural_num num4 = num1 * num2;
  std::cout << num4;
  plural_num num5 = num1 * 2;
  std::cout << num5;
}
void test02() {
    three_d t1(1, 2, 3);
    three_d t2(4, 5, 6);
    three_d t3(7, 8, 9);
    std::cout << t1 << std::endl;
    // ++t1; 
    std::cout << t1 << std::endl; // 显示递增后的 t1
    three_d t6 = t1++; // 使用后缀递增
    std::cout << t6 << std::endl; // 显示递增前的 t1
    std::cout << t1 << std::endl; // 显示递增后的 t1
}
class Date;
class Time{
  
  public:
    Time(int h, int m, int s) : hour_(h), minute_(m), second_(s){};
    void print(Date& d) {
        std::cout << hour_ << ":" << minute_ << ":" << second_ <<d.year_ << "/" << d.month_ << "/" << d.day_ << std::endl;
    }
    private:
        int hour_;
        int minute_;
        int second_;
};
class Date{
  friend void Time::print(Date& d);
  public:

    Date(int y, int m, int d) : year_(y), month_(m), day_(d){};
    private:
        int year_;
        int month_;
        int day_;
};


int main() {
  test01();
  test02();
  return 0;
}
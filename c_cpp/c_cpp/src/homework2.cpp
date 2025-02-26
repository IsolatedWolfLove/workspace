#include <iostream>
#include <string>
class data {
public:
  data(std::string y, std::string m, std::string d) {
    std::cout << "this is data constructor\n";
    year = y;
    month = m;
    day = d;
  }
  std::string get_year() { return year; }
  std::string get_month() { return month; }
  std::string get_day() { return day; }
  std::string get_date() { return year + "-" + month + "-" + day; }
  bool is_leap() {
    int year_int = std::stoi(year);
    int i = 0;
    if (year_int % 4 == 0 && year_int % 100 != 0) {
      i = 1;
    }
    if (year_int % 100 == 0) {
      i = 1;
    }
    return i;
  }

private:
  std::string year;
  std::string month;
  std::string day;
};
class func_test {
  friend void test();
  friend int main();

public:
  class func_test f(func_test other) { return other; }
  func_test() {
    std::cout << "this is normal constructed function\n";
    copy_times++;
  }
  func_test(int id) {
    std::cout << "this is int constructed function\n";
    copy_times++;
    this->id = id;
  }
  func_test(const func_test &other) {
    std::cout << "this is copy constructed function\n";
    this->id = other.id;
    copy_times++;
  }
  ~func_test() {
    std::cout << "this is destructed function\n";
    copy_times--;
  }

private:
  static int copy_times;
  int id;
};
int func_test::copy_times = 0;
void test() {
  func_test f1(1), f3(2);
  func_test f2(f1);
  func_test f4 = f2.f(f1);
  std::cout << "copy_times: " << func_test::copy_times << std::endl;
  std::cout << "f1.id: " << f1.id << std::endl;
  std::cout << "f2.id: " << f2.id << std::endl;
  std::cout << "f3.id: " << f3.id << std::endl;
  std::cout << "f4.id: " << f4.id << std::endl;
}
/*
int main(){
    test();
    std::cout<<func_test::copy_times<<std::endl;
    std::cout<<"------------------------------------------\n";
    func_test* f = new func_test(3);
    delete f;
    return 0;
}*/
int main() {
  data d("2021", "05", "15");
  std::cout << "date: " << d.get_date() << std::endl;
  std::cout << "is leap year: " << d.is_leap() << std::endl;
  return 0;
}
#include "iostream"
#include<string>
#include<algorithm>
// class three_d{
//     public:
//     three_d& operator++(){
//         x_++;
//         y_++;
//         z_++;
//         return *this;
//     };
//     three_d(int x, int y, int z) : x_(x), y_(y), z_(z){};
// void print(){
//     std::cout << x_ << "," << y_ << "," << z_ << std::endl;
// }
//     private:
//         int x_;
//         int y_;
//         int z_;

// };
// class {};
// class {};
// void test01(){

// }
// class marix {
//   friend std::ostream &operator<<(std::ostream &os, const marix &m);

// public:
//   marix(int a[2][3]) {
//     for (int i = 0; i < 2; i++) {
//       for (int j = 0; j < 3; j++) {
//         this->a[i][j] = a[i][j];
//       }
//     }
//   }

// private:
//   int a[2][3];
// };
// std::ostream &operator<<(std::ostream &os, const marix &m) {
//   for (int i = 0; i < 2; i++) {
//     for (int j = 0; j < 3; j++) {
//       os << m.a[i][j] << " ";
//     }
//     os << std::endl;
//   }
//   return os;
// }
// //自定义求两个正整数的最大公约数
// int my_gcd(int a,int b){
//     if(a < b){
//         int temp = a;
//         a = b;
//         b = temp;
//     }
//     return a % b == 0 ? b : my_gcd(b, a % b);
// }
//

//自定义求两个正整数的最大公约数和最小公倍数
void print(){
    std::cout << "print" << std::endl;
    std::cout << "print" << std::endl;
    std::cout << "print" << std::endl;
}
void select_sort(std::string *a) {
  for (int i = 0; i < 5; i++) {
    for (int m = 0; m < a[i].size(); m++) {
      for (int n = m + 1; n < a[i].size(); n++) {
        if (a[i][m] > a[i][n]) {
          char temp = a[i][m];
          a[i][m] = a[i][n];
          a[i][n] = temp;
        }
      }
    }
  }
  for (int i = 0; i < 5; i++) {
    std::cout << a[i] << std::endl;
  }
}
void test01(){
    std::string a[5] = {"mkdrnfnfitrnwuugtwururhfuyhwrfwrhfyrhfhrwyghyrtgyht", "abce", "abcf", "abca", "acbdcefhguwundnedewufn"};
    std::sort(a[0].begin(),a[0].end());
    std::sort(a[1].begin(),a[1].end());
    std::sort(a[2].begin(),a[2].end());
    std::sort(a[3].begin(),a[3].end());
    std::sort(a[4].begin(),a[4].end());
    for(int i = 0; i < 5; i++){
        std::cout << a[i] << std::endl;
    }
}
int main() {
  std::string a[5] = {"mkdrnfnfitrnwuugtwururhfuyhwrfwrhfyrhfhrwyghyrtgyht", "abce", "abcf", "abca", "acbdcefhguwundnedewufn"};
  select_sort(a);
  test01();
  // three_d t1(1, 2, 3);
  // ++t1;
  // t1.print();
  //   int a[2][3] = {1, 2, 3, 4, 5, 6};
  //   marix m1(a);
  //   std::cout << m1 << std::endl;
  //   int a = 33;
  //   int b = 11;
  //   std::cout << my_gcd(a, b) << std::endl;
  return 0;
}
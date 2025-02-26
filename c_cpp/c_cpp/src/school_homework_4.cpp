#include <iostream>
#include <ostream>
#include <string>
#include <vector>
class matix {
public:
  int row, col;

  std::vector<std::vector<int>> data;

  matix(int r, int c, std::initializer_list<std::initializer_list<int>> list)
      : row(r), col(c) {
    data.resize(r, std::vector<int>(c));
    int i = 0, j = 0;
    for (auto &sublist : list) {
      for (auto &elem : sublist) {
        data[i][j] = elem;
        j++;
      }
      i++;
      j = 0;
    }
  }
  matix(std::vector<std::vector<int>> &vec) {
    data = vec;
    vec.swap(data);
  }
  matix() {}
  matix &operator+(const matix &other) {
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        this->data[i][j] += other.data[i][j];
      }
    }
    return *this;
  }
};
std::ostream &operator<<(std::ostream &os, const matix &m) {
  for (int i = 0; i < m.row; i++) {
    for (int j = 0; j < m.col; j++) {
      os << m.data[i][j] << " ";
    }
    os << std::endl;
  }
  return os;
}

int main() {
  matix m1(2, 3, {{1, 2, 3}, {4, 5, 6}});
  matix m2(2, 3, {{7, 8, 9}, {10, 11, 12}});
  matix m3 = m1 + m2;
  std::cout << m3;
  return 0;
}
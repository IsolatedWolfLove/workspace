#include <array>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
std::string change_10_16(int num) {
  static std::string word_16[16] = {"0", "1", "2", "3", "4", "5", "6", "7",
                                    "8", "9", "A", "B", "C", "D", "E", "F"};
  std::vector<int> num_bits;
  std::string result = "0x";
  if (num < 16) {
    result = word_16[num];
  } else {
    while (num > 0) {
      num_bits.push_back(num % 16);
      num /= 16;
    }
  }
  for (int i = num_bits.size() - 1; i >= 0; i--) {
    result += word_16[num_bits[i]];
  }
  return result;
}
std::pair<int, int> searcher_max_3_3(int b[3][3]) {
  int max_mum = 0;
  std::pair<int, int> max_pos;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (b[i][j] > max_mum) {
        max_mum = b[i][j];
        max_pos = std::make_pair(i, j);
      }
    }
  }
  return max_pos;
}
int *searcher_minus_3(int a[3][4]) {
  static int result[3] = {0, 0, 0};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      if (a[i][j] < 0) {
        for (int k = 0; k < 3; k++) {
          result[k] = a[i][k];
        }
      }
    }
  }
  return result;
}
int main(int, char **) {
  std::cout << change_10_16(19876) << std::endl;   // output: 0x0A
  std::cout << change_10_16(1987699) << std::endl; // output: 0x0A
  int a[3][4] = {5, 7, 9, 10, -1, 4, 6, 8, 1, 2, 3, 4};
  int *b = searcher_minus_3(a);
  std::cout << b[0] << " " << b[1] << " " << b[2] << std::endl;
  int c[3][3] = {{5, 7, 9}, {33, 1, 8}, {7, 22, 45}};
  std::pair<int, int> max_pos = searcher_max_3_3(c);
  std::cout << "Max position: (" << max_pos.first << ", " << max_pos.second
            << ")" << std::endl;
  return 0;
}

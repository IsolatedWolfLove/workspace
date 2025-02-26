#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>

#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
class DataStraightRectCalculator {
public:
  DataStraightRectCalculator(std::string file_path) {
    // box_h.resize(256);
    // box_s.resize(256);
    // box_v.resize(256);
    file.open(file_path);
    if (file.is_open()) {
      std::cout << "File opened successfully";
      std::string line;
      while (std::getline(file, line)) {
        std::string sub_line1 = line.substr(0, line.find("  "));
        std::string sub_line2 = line.substr(
            line.find("  ") + 2, line.find("  ", line.find("  ") + 1));
        std::string sub_line3 = line.substr(
            line.find("  ", line.find("  ") + 1) + 2,
            line.find("  ", line.find("  ", line.find("  ") + 1) + 1));
        h = std::stoi(sub_line1);
        s = std::stoi(sub_line2);
        v = std::stoi(sub_line3);
        if (the_first_timer == 0) {
          box_h.push_back(std::make_pair(h, 0));
          box_s.push_back(std::make_pair(s, 0));
          box_v.push_back(std::make_pair(v, 0));
          the_first_timer = 1;
        }
        for (int i = 0; i < box_h.size(); i++) {
          if (h == box_h[i].first) {
            box_h[i].second++;
          } else if (i == box_h.size() - 1) {
            box_h.push_back(std::make_pair(h, 1));
          }
          if (s == box_s[i].first) {
            box_s[i].second++;
          } else if (i == box_h.size() - 1) {
            box_s.push_back(std::make_pair(s, 1));
          }
          if (v == box_v[i].first) {
            box_v[i].second++;
          } else if (i == box_h.size() - 1) {
            box_v.push_back(std::make_pair(v, 1));
          }
        }
      }
    }
    // else {
    //   std::cout << "Could not open file";
    //   return;
    // }
  }

  std::vector<std::pair<int, int>>
  print_the_number_of_the_max_value(int number) {
    std::vector<std::pair<int, int>> the_max_value;
    int count_h = 0;
    for (int i = 0; i < box_h.size(); i++) {
      if (count_h < number) {
        the_max_value.push_back(box_h[i]);
        count_h++;
      } else {
        for (int j = 0; j < number; j++) {
          if (box_h[i].second > the_max_value[j].second) {
            the_max_value[j] = box_h[i];
          }
        }
      }
    }
    return the_max_value;
  }
  ~DataStraightRectCalculator() {
    file.close();
    std::cout << "File closed successfully";
  }

private:
  bool the_first_timer = 0;
  std::ifstream file;
  int h = 0;
  int s = 0;
  int v = 0;
  std::vector<std::pair<int, int>> box_h;
  std::vector<std::pair<int, int>> box_s;
  std::vector<std::pair<int, int>> box_v;
};
int main() {
  DataStraightRectCalculator calculator("/home/ccy/workspace/c_cpp/face.txt");
  std::vector<std::pair<int, int>> the_max_value =
      calculator.print_the_number_of_the_max_value(5);
      for (int i = 0; i < the_max_value.size(); i++) {
        std::cout << "The " << i + 1 << "th max value is " <<
        the_max_value[i].first
                  << " with " << the_max_value[i].second << " times" <<
                  std::endl;
      }
  //2
  return 0;
}
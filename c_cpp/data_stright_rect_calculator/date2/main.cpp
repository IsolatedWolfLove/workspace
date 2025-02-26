#include <fstream>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>
class DataStraightRectCalculator {
public:
  DataStraightRectCalculator(const std::string picture_path) {
    std::fstream file("/home/ccy/workspace/date2/"+picture_path + ".txt", std::ios::app);

    cv::Mat img = cv::imread(picture_path);
    if (img.empty()) {
      std::cout << "Could not open or find the image!\n";
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2HSV);
    for (int i = 0; i < img.rows; i++) {
      for (int j = 0; j < img.cols; j++) {
        cv::Vec3b &pixel = img.at<cv::Vec3b>(i, j);
        int h = pixel[0];
        int s = pixel[1];
        int v = pixel[2];
        if (s >= 2) {
          file << h << "  " << s << "  " << v << std::endl;
        }
      }
    }
    std::string file_path = picture_path + ".txt";
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
    } else {
      std::cout << "Could not open file";
      return;
    }
  }

  ~DataStraightRectCalculator() {
    file.close();
    std::cout << "File closed successfully";
  }
  void draw_rectangle(int num) {
    std::sort(this->box_h.begin(), this->box_h.end(),
              [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                return a.first > b.first;
              });
    std::sort(this->box_s.begin(), this->box_s.end(),
              [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                return a.first > b.first;
              });
    std::sort(this->box_v.begin(), this->box_v.end(),
              [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                return a.first > b.first;
              });
for(int i=0;i<num;i++){
  std::cout<<i<<std::endl;
  std::cout<<"h:"<<this->box_h[i].first<<" "<<this->box_h[i].second;
  std::cout<<"s:"<<this->box_s[i].first<<" "<<this->box_s[i].second;
  std::cout<<"v:"<<this->box_v[i].first<<" "<<this->box_v[i].second<<std::endl;
}

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
  DataStraightRectCalculator calculator("/home/ccy/workspace/opencv/workspace/opencv_learning/face.png");
  calculator.draw_rectangle(10);
  return 0;
}

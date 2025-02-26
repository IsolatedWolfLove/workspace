#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/opencv.hpp>
int main() {
  cv::Mat img = cv::imread("/home/ccy/workspace/opencv_learning/out/build/"
                           "Clang 14.0.0 x86_64-pc-linux-gnu/picture.jpg");
  if(img.empty()) {
    std::cout << "Failed to load image." << std::endl;
    return -1;
  }
  cv::Vec3b intensity = img.at<cv::Vec3b>(100, 100);
  std::cout << "The intensity of the pixel at (100, 100) is: " << static_cast<int>(intensity[0])
            << std::endl;
  
  return 0;
}
#include <fstream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
int main() {
  std::fstream file("noway_clothes.txt", std::ios::app);

  cv::Mat img = cv::imread(
      "/home/ccy/workspace/opencv_learning/屏幕截图 2024-11-29 163322.png");
  if (img.empty()) {
    std::cout << "Could not open or find the image!\n";
    return -1;
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
  // cv::imshow("Image", img);
  cv::waitKey(10);
  return 0;
}
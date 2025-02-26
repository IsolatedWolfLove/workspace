#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/videoio.hpp>
int main() {
  cv::VideoCapture cap(
      "/home/ccy/workspace/finall_test/data/blue_armor_test.mp4");
  if (!cap.isOpened()) {
    std::cout << "Cannot open the camera" << std::endl;
    return -1;
  }
  cv::Mat frame;
  while (cap.read(frame)) {

    cv::imshow("camera", frame);
  }
  cap.release();
  cv::destroyAllWindows();
  return 0;
}
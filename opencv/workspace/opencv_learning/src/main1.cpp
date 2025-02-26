#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <ostream>
#include <vector>
// int operator<<(std::fstream &file, cv::Mat m) {
// file<<m.data;
// }
int main(int, char **) {
  // std::fstream file;
  // file.open("data1.txt", std::ios::app);
  // std::cout << "Hello, from opencv_learning!\n";
  std::vector<cv::Mat> channels;

  auto img = cv::imread("/home/ccy/workspace/opencv_learning/picture.jpg", 1);
  // cv::Mat hsv;
  std::cout << img.type() << "\n";
  // cv::Rct
      //////// cv::resize();
      cv::split(img, channels);
  cv::add(channels[0],channels[1],channels[2]);
  cv::imshow("b", channels[0]);
  cv::imshow("g", channels[1]);
  cv::imshow("r", channels[2]);
  // cv::cvtColor(img, hsv, cv::COLOR_BGR2GRAY);
  // cv::namedWindow("hsv", cv::WINDOW_AUTOSIZE);
  // cv::imshow("hsv", hsv);
  // cv::namedWindow("test", cv::WINDOW_AUTOSIZE);
  cv::imshow("test", img);
  // cv::imwrite("picture.jpg", img);
  // file<<img;
  //   file.close();
  int key = cv::waitKey();
  ////std::cout<<key<<std::endl;
  ////
  cv::destroyAllWindows();
  // cv::Mat f;
  // cv::VideoCapture(0);
  return 0;
}

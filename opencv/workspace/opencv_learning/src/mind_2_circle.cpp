/*#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
std::vector<cv::Rect> inrang_blue(cv::Mat src, double scale) {
  cv::Mat hsv_image;
  cv::cvtColor(src, hsv_image, cv::COLOR_BGR2HSV);
  cv::Scalar lowerBound(100, 50, 120);
  cv::Scalar upperBound(115, 265, 254);
  cv::Mat mask;
  cv::inRange(hsv_image, lowerBound, upperBound, mask);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  std::vector<cv::Rect> validRects;
  for (size_t i = 0; i < contours.size(); i++) {
    double area = cv::contourArea(contours[i]);
    if (area > 500) {
      cv::Rect rect = cv::boundingRect(contours[i]);
      rect.x *= (1.0 / scale);
      rect.y *= (1.0 / scale);
      rect.width *= (1.0 / scale);
      rect.height *= (1.0 / scale);
      int padding = 200;
      rect.x -= padding;
      rect.y -= padding;
      rect.width += 2 * padding;
      rect.height += 2 * padding;
      rect &= cv::Rect(0, 0, src.cols, src.rows);
      validRects.push_back(rect);

    }
  }
  return validRects;
}
int cou;
void circle_detector_draw_rect(cv::Mat src_, cv::Mat src,
                               std::vector<cv::Rect> validRects, double scale) {
  cv::Mat grayImage;
  cv::cvtColor(src, grayImage, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(grayImage, grayImage, cv::Size(3, 3), 3);
  // 应用Canny边缘检测
  cv::Mat edges;
  double threshold1 = 10;  // 低阈值
  double threshold2 = 100; // 高阈值
  cv::Canny(grayImage, edges, threshold1, threshold2);
  std::vector<cv::Vec3f> circles;
  // 设置Hough变换参数
  int minDist = 50; // 最小距离
  int param1 = 100; // Canny边缘检测高阈值  # 保持和和前面Canny的阈值一致
  int param2 = 28;    // Hough变换高阈值
  int minRadius = 35; // 最小半径
  int maxRadius = 55; // 最大半径
  HoughCircles(edges, circles, cv::HOUGH_GRADIENT, 1, minDist, param1, param2,
               minRadius, maxRadius);
  //在原图上绘制检测到的圆

  for (size_t i = 0; i < circles.size(); i++) {
    cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    std::cout << circles.size() << std::endl;
    int radius = cvRound(circles[i][2]);
    // if(radius < 30){continue;}
    circle(src_, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0); // 绘制圆心
    circle(src_, center, radius + 50, cv::Scalar(0, 0, 255), 3, 8, 0);
    for (const auto &rect : validRects) {
      if (rect.contains(center)) {
        rectangle(src_, rect, cv::Scalar(0, 255, 0), 2);
cou++;
        break;
      }
    }
  }
  cv::resize(src_, src_, cv::Size(src.cols / 6, src.rows / 6), 0, 0,
             cv::INTER_LINEAR);
  cv::imshow("src_", src_);
}
int main(int argc, char **argv) {
  cv::Mat src_ =
      cv::imread("/home/ccy/workspace/opencv_learning/src/airconditionns.jpg");
  cv::Mat src;
  double scale = 2.0;
  int newWidth = static_cast<int>(src_.cols * scale);
  int newHeight = static_cast<int>(src_.rows * scale);
  cv::resize(src_, src, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
  std::vector<cv::Rect> rect = inrang_blue(src, scale);
  circle_detector_draw_rect(src_, src,rect, scale);
  std::cout<<"cou:"<<cou<<std::endl;
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}*/
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  cv::Mat src_ =
      cv::imread("/home/ccy/workspace/opencv_learning/src/airconditionns.jpg");
  cv::Mat src;
  double scale = 2.0;
  int newWidth = static_cast<int>(src_.cols * scale);
  int newHeight = static_cast<int>(src_.rows * scale);
  cv::resize(src_, src, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
  cv::Mat grayImage;
  cv::cvtColor(src, grayImage, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(grayImage, grayImage, cv::Size(3, 3), 3);
  // 应用Canny边缘检测
  cv::Mat edges;
  double threshold1 = 10;  // 低阈值
  double threshold2 = 100; // 高阈值
  cv::Canny(grayImage, edges, threshold1, threshold2);
  std::vector<cv::Vec3f> circles;
  // 设置Hough变换参数
  int minDist = 50; // 最小距离
  int param1 = 100; // Canny边缘检测高阈值  # 保持和和前面Canny的阈值一致
  int param2 = 32;    // Hough变换高阈值
  int minRadius = 35; // 最小半径
  int maxRadius = 55; // 最大半径
  HoughCircles(edges, circles, cv::HOUGH_GRADIENT, 1, minDist, param1, param2,
               minRadius, maxRadius);
  // 在原图上绘制检测到的圆
  for (size_t i = 0; i < circles.size(); i++) {
    cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    int radius = cvRound(circles[i][2]);
    // if(radius < 30){continue;}
    circle(src, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0); // 绘制圆心
    circle(src, center, radius + 50, cv::Scalar(0, 0, 255), 3, 8, 0);
    //绘制空心圆
  }
  cv::resize(edges, edges, cv::Size(edges.cols / 4, edges.rows / 4), 0, 0,
             cv::INTER_LINEAR);
  cv::resize(src, src, cv::Size(src.cols / 6, src.rows / 6), 0, 0,
             cv::INTER_LINEAR);
  cv::imshow("src", src);
  cv::imshow("edges", edges);

  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}

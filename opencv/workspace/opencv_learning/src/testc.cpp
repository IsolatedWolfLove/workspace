#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
int main() {
  cv::Mat src_ =
      cv::imread("/home/ccy/workspace/opencv_learning/src/airconditionns.jpg");
  cv::Mat src;
  double scale = 2.0;
  int newWidth = static_cast<int>(src_.cols * scale);
  int newHeight = static_cast<int>(src_.rows * scale);
  cv::resize(src_, src, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
  cv::Mat hsv_image;
  cv::cvtColor(src, hsv_image, cv::COLOR_BGR2HSV);
  cv::Scalar lowerBound(100, 50, 120);
  cv::Scalar upperBound(115, 265, 254);
  cv::Mat mask;
  cv::inRange(hsv_image, lowerBound, upperBound, mask);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  for (size_t i = 0; i < contours.size(); i++) {
    double area = cv::contourArea(contours[i]);
    if (area > 500) {
      cv::Rect rect = cv::boundingRect(contours[i]);
      rect.x *= (1.0 / scale);
      rect.y *= (1.0 / scale);
      rect.width *= (1.0 / scale);
      rect.height *= (1.0 / scale);
      int padding = 50;
      rect.x -= padding;
      rect.y -= padding;
      rect.width += 2 * padding;
      rect.height += 2 * padding;
      rect &= cv::Rect(0, 0, src.cols, src.rows);
      cv::rectangle(src_, rect, cv::Scalar(0, 255, 0), 2);
    }
  }
  cv::imshow("src_", src_);
  cv::Mat dst;
  cv::resize(src_, dst, cv::Size(1280, 640), 0.5, 0.5, cv::INTER_LINEAR);
  cv::imshow("dst", dst);
  cv::imshow("src", src);
  cv::imshow("mask", mask);
  cv::Mat mask_re;
  cv::resize(mask, mask_re, cv::Size(1280, 640), 0.5, 0.5, cv::INTER_LINEAR);
  cv::imshow("mask_re", mask_re);
  cv::imshow("hsv_image", hsv_image);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}
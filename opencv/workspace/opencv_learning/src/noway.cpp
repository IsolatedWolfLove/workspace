#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
int main() {
  cv::Mat frame =
      cv::imread("/home/ccy/workspace/opencv_learning/src/airconditionns.jpg");
  cv::resize(frame, frame, cv::Size(1280, 960));
  cv::Mat hsv_image;
  // cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
  // cv::filter2D(frame, frame, frame.depth(), kernel);
  //转化为hsv空间
  cv::cvtColor(frame, hsv_image, cv::COLOR_BGR2HSV);
  //设置颜色范围
  cv::Scalar lowerBound(100, 50, 120);
  cv::Scalar upperBound(115, 265, 254);
  cv::Mat mask;
  cv::inRange(hsv_image, lowerBound, upperBound, mask);
  //寻找轮廓
  cv::Mat contour_image;
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  for (size_t i = 0; i < contours.size(); i++) {
    double area = cv::contourArea(contours[i]);
    if (1) {
      cv::Rect boundingRect = cv::boundingRect(contours[i]);
      cv::rectangle(frame, boundingRect, cv::Scalar(0, 0, 255), 9);
      break;
    }
  }
  frame.copyTo(contour_image);
  cv::drawContours(contour_image, contours, -1, cv::Scalar(0, 255, 0), 2);
  cv::imshow("Contours", contour_image);
  cv::imshow("mask", mask);
  cv::imshow("frame", frame);
cv::waitKey(0);
  return 0;
}
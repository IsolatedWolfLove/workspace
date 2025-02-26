#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
std::vector<cv::Point> circle_detector(const cv::Mat src) {
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
  std::vector<cv::Point> centers;
  HoughCircles(edges, circles, cv::HOUGH_GRADIENT, 1, minDist, param1, param2,
               minRadius, maxRadius);
  // 在原图上绘制检测到的圆
  for (size_t i = 0; i < circles.size(); i++) {
    cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    int radius = cvRound(circles[i][2]);
    // if(radius < 30){continue;}
    // circle(src, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0); // 绘制圆心
    // circle(src, center, radius + 50, cv::Scalar(0, 0, 255), 3, 8, 0);
    centers.push_back(center);
    //绘制空心圆
  }
  // cv::resize(src, src, cv::Size(src.cols / 8, src.rows / 8), 0.5, 0.5,
  //            cv::INTER_LINEAR);
  // cv::imshow("src", src);
  return centers;
}
void inrange_bule_detect(cv::Mat src_, cv::Mat src,
                         std::vector<cv::Point> centers, double scale) {
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
      int padding = 100;
      rect.x -= padding;
      rect.y -= padding;
      rect.width += 2 * padding;
      rect.height += 2 * padding;
      for (cv::Point center : centers) {
        if (rect.contains(center)) {
          rect.x *= (1.0 / scale);
          rect.y *= (1.0 / scale);
          rect.width *= (1.0 / scale);
          rect.height *= (1.0 / scale);
          rect &= cv::Rect(0, 0, src.cols, src.rows);
          cv::rectangle(src_, rect, cv::Scalar(0, 0, 255), 2);
        }
      }
    }
  }
  cv::imshow("src_", src_);
  cv::Mat dst;
  cv::resize(src_, dst, cv::Size(1280, 640), 0.5, 0.5, cv::INTER_LINEAR);
  cv::imshow("dst", dst);
  cv::waitKey(0);
  cv::destroyAllWindows();
}
int main() {
  cv::Mat src_ =
      cv::imread("/home/ccy/workspace/opencv/workspace/opencv_learning/src/airconditionns.jpg");
  cv::Mat src;
  double scale = 2.0;
  int newWidth = static_cast<int>(src_.cols * scale);
  int newHeight = static_cast<int>(src_.rows * scale);
  cv::resize(src_, src, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
  std::vector<cv::Point> centers = circle_detector(src);
  inrange_bule_detect(src_, src, centers, scale);
  return 0;
}

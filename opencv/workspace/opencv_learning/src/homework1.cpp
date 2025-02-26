#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
int main() {
  int min_area = 8200;
  cv::VideoCapture cap("/home/ccy/Downloads/test.mp4");
  if (!cap.isOpened()) {
    std::cout << "Cannot open the video file" << std::endl;
    return -1;
  }
  cv::Mat frame;
  while (1) {
    // double begin_time = (double)cv::getTickCount();
    cap.read(frame);
    cv::Mat hsv_image;
    // cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    // cv::filter2D(frame, frame, frame.depth(), kernel);
    //转化为hsv空间
    cv::cvtColor(frame, hsv_image, cv::COLOR_BGR2HSV);
    //设置颜色范围
    cv::Scalar lowerBound(0, 29, 139); 
    cv::Scalar upperBound(11, 171, 248); 
    cv::Mat mask;
    cv::inRange(hsv_image, lowerBound, upperBound, mask);
    //寻找轮廓
    cv::Mat contour_image;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++) {
      double area = cv::contourArea(contours[i]);
      if (area > min_area) { 
        cv::Rect boundingRect = cv::boundingRect(contours[i]);
        cv::rectangle(frame, boundingRect, cv::Scalar(0, 0, 0),
                      5); 
        break;
      }
    }
    frame.copyTo(contour_image);
    cv::drawContours(contour_image, contours, -1, cv::Scalar(0, 255, 0), 2);
    cv::imshow("Contours", contour_image);
    cv::imshow("mask", mask);
    cv::imshow("frame", frame);
    // double end_time = (double)cv::getTickCount();
    // double time_used = (end_time - begin_time) / cv::getTickFrequency();
    // std::cout << "Time used: " << time_used << " s" << std::endl;
    // std::cout << "isContinuous: " << frame.isContinuous();
    int key = cv::waitKey(25);
    if (key == 27) {
      break;
    }
  }
  cap.release();
  cv::destroyAllWindows();
  return 0;
}
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
std::vector<cv::Point2f> detectValidCorners(const cv::Mat &image, int blockSize,
                                          int minCorners) {
  std::vector<cv::Point2f> validCorners; // 用于存储有效角点

  // 获取图像尺寸
  int imgRows = image.rows;
  int imgCols = image.cols;

  // 遍历图像的每个小矩形区域
  for (int row = 0; row < imgRows; row += blockSize) {
    for (int col = 0; col < imgCols; col += blockSize) {
      cv::Mat subMat =
          image(cv::Range(row, std::min(row + blockSize, imgRows)),
                cv::Range(col, std::min(col + blockSize, imgCols)));

      // Shi-Tomasi角点检测
      std::vector<cv::Point> cornersInSubMat;
      goodFeaturesToTrack(subMat, cornersInSubMat, 1000, 0.1, 10);

      // 检查角点数量是否少于阈值
      if ((int)cornersInSubMat.size() < minCorners) {
        // 将有效角点的坐标调整为原图坐标
        for (auto &corner : cornersInSubMat) {
          corner += cv::Point(col, row);
          validCorners.push_back(corner); // 将角点添加到validCorners中
        }
      }
    }
  }

  return validCorners;
}
int main() {
  cv::Mat src =
      cv::imread("/home/ccy/workspace/opencv_learning/src/airconditionns.jpg");

  double scale = 0.5;
  int newWidth = static_cast<int>(src.cols * scale);
  int newHeight = static_cast<int>(src.rows * scale);

  //使用cv::resize函数来缩小图片
  cv::resize(src, src, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
  cv::Mat before_filter_gray_src, gray_src, grad_x, grad_y, canny_dst;
  cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray_src, gray_src, cv::Size(7, 7), 7);
  // cv::bilateralFilter(before_filter_gray_src, gray_src, 50, 50, 100);
  // blur(gray_src, gray_src, cv::Size(3,3));

  /*
  cv::Sobel(gray_src, grad_x, CV_16S, 1, 0,3,  1,5);
  cv::Sobel(gray_src, grad_y, CV_16S, 0, 1, 3, 1, 5);
  cv::convertScaleAbs(grad_x, grad_x);
  cv::convertScaleAbs(grad_y, grad_y);*/
  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::erode(gray_src, gray_src, element);
  cv::dilate(gray_src, gray_src, element);
  cv::Canny(gray_src, canny_dst, 25, 50, 3, true);

  std::vector<cv::Point2f> corners = detectValidCorners(canny_dst,50,5)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       s(canny_dst,25,100);
  // 在原图上绘制检测到的角点
  for (cv::Point2f corner : corners) {
    cv::circle(src, corner, 5, cv::Scalar(255, 0, 0), -1);
  }
  /*
    std::vector<cv::Vec3f> circles;
    // 设置Hough变换参数
    int minDist = 50; // 最小距离
    int param1 = 50; // Canny边缘检测高阈值  # 保持和和前面Canny的阈值一致
    int param2 = 50;    // Hough变换高阈值
    int minRadius = 10; // 最小半径
    int maxRadius = 50; // 最大半径
    HoughCircles(canny_dst, circles, cv::HOUGH_GRADIENT, 1, minDist, param1,
                 param2, minRadius, maxRadius);
    // 在原图上绘制检测到的圆
    for (size_t i = 0; i < circles.size(); i++) {
      cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      circle(src, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);     // 绘制圆心
      circle(src, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
    //绘制空心圆
    }
  */
  cv::imshow("src", src);
  cv::imshow("gray_src", gray_src);
  /*
  cv::imshow("grad_x", grad_x);
  cv::imshow("grad_y", grad_y);*/
  cv::imshow("canny_dst", canny_dst);
  cv::waitKey(0);
  return 0;
}
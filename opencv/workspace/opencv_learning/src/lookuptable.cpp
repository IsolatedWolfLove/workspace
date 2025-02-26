#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
int main() {
  cv::Mat src = cv::imread("/home/ccy/workspace/opencv_learning/out/build/"
                           "Clang 14.0.0 x86_64-pc-linux-gnu/picture.jpg");
  if(src.empty()) {
    std::cerr << "Could not open or find the image!\n";
    return -1;
  }
  /*
  cv::Mat lookUpTable(1, 256, CV_8UC1);
  int threshold = 128;
  for(int i = 0; i < 256; i++) {
    if(i < threshold) {
      lookUpTable.at<uchar>(0, i) = 0;
    } else {
      lookUpTable.at<uchar>(0, i) = 255;
    }
  }*/
  // cv::Mat dst;
  // cv::LUT(src, lookUpTable, dst);
  cv::Mat kernel_result;
  cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
  cv::filter2D(src, src, CV_8UC1, kernel);
  cv::imshow("src", src);
  // cv::imshow("dst", dst);
  // cv::imshow("kernel_result", kernel_result);
  cv::waitKey(0);
  return 0;
}
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
void key_Demo(cv::Mat& img) {
	while(true) {
		int k = waitKey(0);
		std::cout << k << std::endl;
        cv::imshow("image", img);
	}
}
int main() {
    Mat img = Mat(100, 100, CV_8UC3, Scalar(0, 0, 0));
    key_Demo(img);
    return 0;
}
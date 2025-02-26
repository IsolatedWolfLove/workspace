#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

class ImageClassifier {
public:
  ImageClassifier(std::string &model_path_) {
    picture_size = std::make_pair(default_size, default_size);
    model_path = model_path_;
    load_model();
  }
  void classify_image(cv::Mat &image) {
    this->image = image;
    cv::Mat input_tensor = picture_fit_model();

    net.setInput(input_tensor);
    cv::Mat output = net.forward();
    std::vector<float> output_vec = output.reshape(1, 5).col(0);
    int max_index = 0;
    float max_value = output_vec[0];
    for (int i =0; i < 5; ++i) {
      std::cout << "class " << i << " : " << output_vec[i] << std::endl;
      if (output_vec[i] > max_value) {
        max_index = i;
        max_value = output_vec[i];
      }
    }
    std::cout << "The image is classified as: " << max_index << std::endl;
    // if(words[1] == 'daisy'):
    //               words[1] = 0
    //           elif(words[1] == 'dandelion'):
    //               words[1] = 1
    //           elif(words[1] == 'rose'):
    //               words[1] = 2
    //           elif(words[1] == 'sunflower'):
    //               words[1] = 3
    //           elif(words[1] == 'tulip'):
    //               words[1] = 4
    if (max_index == 0) {
      std::cout << "The image is a daisy" << std::endl;
    } else if (max_index == 1) {
      std::cout << "The image is a dandelion" << std::endl;
    } else if (max_index == 2) {
      std::cout << "The image is a rose" << std::endl;
    } else if (max_index == 3) {
      std::cout << "The image is a sunflower" << std::endl;
    } else if (max_index == 4) {
      std::cout << "The image is a tulip" << std::endl;
    }
  }

private:
  int default_pici = 100;
  int default_size = 28;
  std::pair<int, int> picture_size;
  std::string model_path;
  cv::dnn::Net net;
  cv::Mat image;
  void change_picture_size(int width, int height) {
    picture_size = std::make_pair(width, height);
  }
  void change_picture_pici(int pici) { default_pici = pici; }

  cv::Mat picture_fit_model() {
    // 检查图像是否加载成功
    if (image.empty()) {
      std::cout << "Could not read the image" << std::endl;
      return cv::Mat();
    }
    std::vector<cv::Mat> images;
    images.reserve(default_pici);
    for (int i = 0; i < default_pici; i++) {
      images.push_back(image);
    }
    cv::Mat input_blob;
    input_blob = cv::dnn::blobFromImages(
        images, 1.0, cv::Size(picture_size.first, picture_size.second),
        cv::Scalar(0, 0, 0), false, false);
    return input_blob;
  }
  void load_model() {
    cv::dnn::Net net_ = cv::dnn::readNetFromONNX(model_path);
    // 设置CUDA为后端
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    this->net = net_;
  }
};

int main() {
  std::string model_path =
      "/home/ccy/workspace/deeplearning/flower_cnn_piches100_200.onnx";
  ImageClassifier classifier(model_path);
  cv::Mat image = cv::imread("/home/ccy/workspace/deeplearning/learning_conda/"
                             "src/flower_det/屏幕截图 2024-12-08 221107.png",
                             cv::IMREAD_COLOR);
  classifier.classify_image(image);

  cv::imshow("image", image);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}
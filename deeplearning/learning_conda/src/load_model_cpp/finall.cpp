#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <cmath>
#include <cv_bridge/cv_bridge.h>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>
/*
distence_between_two_points函数返回的是两个点之间的距离
*/
double distence_between_two_points(cv::Point2f &x, cv::Point2f &y) {
  return std::sqrt(std::pow(x.x - y.x, 2) + std::pow(x.y - y.y, 2));
}
/*
ImageClassifier类用于加载和使用模型进行图像分类，主要用于分类数字，
输入图像为灰度图，经过picture_fit_model处理后输出为输入图像的28*28的矩阵
再进入模型进行分类，输出为10个浮点数，表示每个数字的概率，
最后通过classify_image函数返回概率最大的数字
*/
class ImageClassifier {
public:
  ImageClassifier(std::string &model_path_) {
    set_model_path(model_path_);
    picture_size = std::make_pair(default_size, default_size);
    load_model();
  }
  void set_model_path(std::string &model_path_) { model_path = model_path_; }

  std::string classify_image(cv::Mat &image) {
    this->image = image;
    cv::Mat input_tensor = picture_fit_model();
    std::string result_str;
    net.setInput(input_tensor);
    cv::Mat output = net.forward();
    std::vector<float> output_vec = output.reshape(1, 10).col(0);
    int max_index = 0;
    float max_value = output_vec[0];
    for (int i = 1; i < 10; ++i) {
      if (output_vec[i] > max_value) {
        max_index = i;
        max_value = output_vec[i];
      }
    }
    result_str = std::to_string(max_index);
    return result_str;
  }

private:
  int default_pici = 1;
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
      // image = 255 - image;//已经是黑色不再需要了
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
/*
picture_dealor类用于处理图像，主要用于检测装甲板，
输入图像为彩色图，经过color_detect处理后输出为灰度图，
再经过light_bar_detector处理后输出为装甲板的位置和数字，
此为主要的类，包含几乎所有处理函数
*/
class picture_dealor {

public:
  picture_dealor(std::string &model_path_, bool is_enemy_blue_)
      : is_enemy_blue(is_enemy_blue_), classifier(model_path_) {

    cv::Mat k;
    cv::Mat template_image =
        cv::imread("/home/ccy/workspace/finall_test/data/template_image.png",
                   cv::IMREAD_GRAYSCALE);
    this->template_image = template_image;
    if (is_enemy_blue) {
      binary_threshold = 75;
      min_light_bar_area = 1000;
      max_light_bar_area = 13000;
      max_light_bar_width_length_diff = 1.2;
      max_light_bar_angle = 30;
      max_bin_point_y_diff = 0.3;
      max_bin_length_diff = 30;
      bin_diatance_times = 2.3;
    } else {
      binary_threshold = 75;
      min_light_bar_area = 188;
      max_light_bar_area = 8888;
      max_light_bar_width_length_diff = 1.2;
      max_light_bar_angle = 22;
      max_bin_point_y_diff = 0.5;
      max_bin_length_diff = 8;
      bin_diatance_times = 4;
    }
  }
  /*
  color_detect函数用于检测图像中的颜色，
  输入图像为彩色图，经过color_detect处理后输出为灰度图,是对图像根据检测蓝或红分别处理
  */
  cv::Mat color_detect(cv::Mat &image) {
    this->image = image;
    std::vector<cv::Mat> channels;
    //根据bgr分离通道
    cv::split(image, channels);
    cv::Mat gray, binary;
    if (is_enemy_blue) {
      // 蓝色因为杂质较少，所以仅需分离处理即可
      cv::subtract(channels[0] + channels[1] * 0.4, channels[2], gray);
    } else {
      // 红色因为杂质较多，所以需要多步处理
      // 第一步，bgr分离处理，突出红色，但为了保留灯条中心亮白色，同时尽可能减少装甲板数字白色干扰
      //所以我尽可能增加绿色的权重，30就是这么来的，再进行二值化处理
      cv::Mat red = channels[2] * 2 - (channels[0] + channels[1] * 30);
      cv::threshold(red, red, 220, 255, cv::THRESH_BINARY);
      //转化hsv作用是处理明度，即亮度很高
      cv::Mat hsv;
      cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV); // 将BGR图像转换为HSV
      std::vector<cv::Mat> hsv_channels;
      cv::split(hsv, hsv_channels);    // 分离HSV通道
      cv::Mat value = hsv_channels[2]; // V通道 - 明度
      cv::Mat value_mask;
      cv::threshold(value, value_mask, 240, 255, cv::THRESH_BINARY);
      // 将色调mask和饱和度mask进行与运算，得到最终结果,用以达到尽最大可能保住灯带中间白色但不要数字白
      cv::bitwise_and(red, value_mask, gray);
      // 直方图均衡化，增强对比度，减少亮度不均对其的影响
      cv::equalizeHist(gray, gray);
    }
    //二值化
    cv::threshold(gray, binary, 75, 255, cv::THRESH_BINARY);
    cv::GaussianBlur(binary, binary, cv::Size(5, 5),
                     0); // 对通道进行高斯滤波
    // 进行闭操作，缝合灯条中的缝隙
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(5, 5)); // 5x5 的矩形结构元素
    cv::Mat closed;
    cv::morphologyEx(binary, closed, cv::MORPH_CLOSE, kernel);
    //之后交给light_bar_detector函数
    return light_bar_detector(closed);
  }

private:
  // 是否是敌方蓝方
  bool is_enemy_blue = 0;
  // images
  cv::Mat image;
  cv::Mat template_image;
  cv::Mat result;
  //识别数字类
  ImageClassifier classifier;
  // param
  int binary_threshold;
  float min_light_bar_area;
  float max_light_bar_area;
  float max_light_bar_width_length_diff;
  float max_light_bar_angle;
  float max_bin_point_y_diff;
  float max_bin_length_diff;
  double bin_diatance_times;
  //
  // 计算矩形的角度，这里是长边
  // mode=0 表示不区分正负角，mode=1 表示区分正负角
  float angle_deal(cv::RotatedRect &rect, int mode = 0) {
    float angle = rect.angle; // 计算矩形的角度
    if (rect.size.width > rect.size.height) {
      angle = 90 - angle; // 长边与竖直方向的夹角
    } else {
      angle = angle; // 直接使用 angle
    }

    if (angle > 90) {
      if (mode == 0)
        angle = 180 - angle;
      else
        angle = angle - 180;
    }

    return angle;
  }
  //透视变换把装甲板矫正到模板图像角度大小
  cv::Mat correctPerspective(const cv::Mat &input_image,
                             const cv::Mat &template_image) {
    // 目标图像的四个角点（按顺序：左上、右上、右下、左下）
    std::vector<cv::Point2f> dst_corners = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(template_image.cols), 0),
        cv::Point2f(static_cast<float>(template_image.cols),
                    static_cast<float>(template_image.rows)),
        cv::Point2f(0, static_cast<float>(template_image.rows))};

    // 源图像的四个角点
    std::vector<cv::Point2f> src_corners = {
        cv::Point2f(0, 0), cv::Point2f(static_cast<float>(input_image.cols), 0),
        cv::Point2f(static_cast<float>(input_image.cols),
                    static_cast<float>(input_image.rows)),
        cv::Point2f(0, static_cast<float>(input_image.rows))};

    // 计算透视变换矩阵并应用变换
    cv::Mat perspective_matrix =
        cv::getPerspectiveTransform(src_corners, dst_corners);
    cv::Mat output;
    cv::warpPerspective(input_image, output, perspective_matrix,
                        template_image.size());

    return output;
  }
  // 添加模板匹配函数声明，用于判断装甲板是否是数字，因为红色的太多杂色了。。。
  bool template_match(const cv::Mat &target_img, double similarity_threshold) {
    const std::string template_dir = "/home/ccy/workspace/finall_test/data/"
                                     "template"; // 替换为您的模板目录路径
    double best_similarity = 0.0;

    // 确保目标图像是灰度图
    cv::Mat target_gray;
    if (target_img.channels() > 1) {
      cv::cvtColor(target_img, target_gray, cv::COLOR_BGR2GRAY);
    } else {
      target_gray = target_img.clone();
    }

    try {
      for (const auto &entry :
           std::filesystem::directory_iterator(template_dir)) {
        cv::Mat template_img =
            cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (template_img.empty()) {
          std::cerr << "Failed to read template: " << entry.path() << std::endl;
          continue;
        }

        // 将模板调整为与目标图像相同大小
        cv::resize(template_img, template_img, target_gray.size());

        // 计算相似度
        cv::Mat target_norm, template_norm;
        cv::normalize(target_gray, target_norm, 0, 1, cv::NORM_MINMAX);
        cv::normalize(template_img, template_norm, 0, 1, cv::NORM_MINMAX);

        cv::Mat result;
        cv::matchTemplate(target_norm, template_norm, result,
                          cv::TM_CCOEFF_NORMED);
        double similarity;
        cv::minMaxLoc(result, nullptr, &similarity);

        best_similarity = std::max(best_similarity, similarity);

        // 如果已经找到足够相似的模板，可以提前退出
        if (best_similarity >= similarity_threshold) {
          return true;
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "Error during template matching: " << e.what() << std::endl;
      return false;
    }

    return best_similarity >= similarity_threshold;
  }
  // 检测灯条，输入为二值化后的图像，输出为装甲板位置
  cv::Mat light_bar_detector(cv::Mat &image_closed) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image_closed, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE); // 寻找轮廓
    std::vector<cv::RotatedRect> rectangles; // 用于存储符合条件的矩形
    for (size_t i = 0; i < contours.size(); i++) {
      if (!is_enemy_blue) {
        // 因为椭圆拟合的限制，如果轮廓点数小于5，则跳过
        if (contours[i].size() < 5) {
          continue;
        }
        // 计算轮廓面积，用于判断是否是灯条
        double area = cv::contourArea(contours[i]);
        if (area < min_light_bar_area || area > max_light_bar_area) {
          continue;
        }
        //这里用了椭圆拟合，因为矩形拟合可能会有误差
        cv::RotatedRect minRect = cv::fitEllipse(contours[i]);
        cv::Point2f rect_points[4];
        minRect.points(rect_points); // 获取矩形的四个顶点

        // float width = minRect.size.width;
        // float height = minRect.size.height;
        // float max_length = std::max(width, height);
        // float min_length = std::min(width, height);
        float angle = angle_deal(minRect);
        // 判断角度是否在阈值范围内
        if ((std::abs(angle) > max_light_bar_angle)) {
          continue;
        }
        rectangles.push_back(minRect);
      } else {
        // 如果是敌方红方，则要多算一点
        cv::RotatedRect minRect =
            cv::minAreaRect(contours[i]); // 计算最小外接矩形
        cv::Point2f rect_points[4];
        minRect.points(rect_points); // 获取矩形的四个顶点

        float width = minRect.size.width;
        float height = minRect.size.height;
        float area = width * height;
        float max_length = std::max(width, height);
        float min_length = std::min(width, height);
        float angle = angle_deal(minRect);
        // 判断面积是否在阈值范围内
        if ((area < min_light_bar_area) || (area > max_light_bar_area)) {
          continue;
        }
        // 判断长宽差是否在阈值范围内
        if (std::abs(max_length - min_length) <
            max_light_bar_width_length_diff) {
          continue;
        }
        // 判断角度是否在阈值范围内
        if ((std::abs(angle) > max_light_bar_angle)) {
          continue;
        }
        rectangles.push_back(minRect);
      }
    }
    // 检测装甲板
    armour_detect(rectangles);
    return image;
  }
  // 检测装甲板，输入为灯条
  void armour_detect(std::vector<cv::RotatedRect> &rectangles) {
    // 用于存储符合条件的装甲板对
    std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>> suitable_pairs_one;
    // 对装甲板进行排序，按x坐标排序
    std::sort(rectangles.begin(), rectangles.end(),
              [](const cv::RotatedRect &a, const cv::RotatedRect &b) {
                return a.center.x < b.center.x;
              });

    for (size_t i = 0; i < rectangles.size(); i++) {
      for (size_t j = i + 1; j < rectangles.size(); j++) { // 对每对矩形进行比较
        float length_i = std::max(rectangles[i].size.width,
                                  rectangles[i].size.height); // 长边
        float length_j =
            std::max(rectangles[j].size.width, rectangles[j].size.height);
        float avage_length = (length_i + length_j) / 2; // 平均长度
        cv::Point2f points_i[4];
        cv::Point2f points_j[4];
        rectangles[i].points(points_i);
        rectangles[j].points(points_j);

        // 找到每个矩形的左上角点（y坐标最小的点）
        float i_y = points_i[0].y;
        float j_y = points_j[0].y;
        for (int k = 1; k < 4; k++) {
          i_y = std::min(i_y, points_i[k].y);
          j_y = std::min(j_y, points_j[k].y);
        }
        // 判断装甲板对最上角点y坐标差是否在阈值范围内
        if ((std::abs(i_y - j_y) / avage_length) > max_bin_point_y_diff) {
          continue;
        }

        // 判断装甲板对角度差是否在阈值范围内
        if (std::abs(angle_deal(rectangles[i], 1) -
                     angle_deal(rectangles[j], 1)) > max_bin_length_diff) {
          continue;
        }
        if (!is_enemy_blue) {
          // 检查长度是否相近
          if ((std::abs(length_i - length_j)) > avage_length / 6) {
            continue;
          }
        } else {
          if ((std::pow(length_i - avage_length, 2) +
               std::pow(length_j - avage_length, 2)) > max_bin_length_diff) {
            continue;
          }
        }
        // 计算中心点

        cv::Point2f center_i = rectangles[i].center;
        cv::Point2f center_j = rectangles[j].center;

        // 计算中心点之间的距离
        double distance = distence_between_two_points(center_i, center_j);
        double distance_threshold = bin_diatance_times * (avage_length);
        // 检查距离是否在阈值范围内，阈值依据装甲板长度改变而改变
        if (!is_enemy_blue) {
          if (distance > distance_threshold || distance < avage_length) {
            continue;
          }
        } else {
          // 如果是敌方红方，则需要范围大一点
          if (distance > distance_threshold) {
            continue;
          }
        }
        // 储存符合条件的装甲板对
        suitable_pairs_one.push_back(
            std::make_pair(rectangles[i], rectangles[j]));
        // 找到符合条件的装甲板对后，跳出循环
        break;
      }
    }
    /////////////////////////////
    // 检测装甲板，输入为装甲板对
    if (suitable_pairs_one.size() > 0) {
      for (auto &suits_pair : suitable_pairs_one) {
        //再次检查装甲板对，确保装甲板对是正确的
        cv::RotatedRect leftRect =
            suits_pair.first.center.x < suits_pair.second.center.x
                ? suits_pair.first
                : suits_pair.second;
        cv::RotatedRect rightRect =
            suits_pair.first.center.x > suits_pair.second.center.x
                ? suits_pair.first
                : suits_pair.second;
        // 计算装甲板对的长和宽，这里的长是长边
        float left_height = leftRect.size.height > leftRect.size.width
                                ? leftRect.size.height
                                : leftRect.size.width;
        float right_height = rightRect.size.height > rightRect.size.width
                                 ? rightRect.size.height
                                 : rightRect.size.width;
        float left_width = leftRect.size.width > leftRect.size.height
                               ? leftRect.size.width
                               : leftRect.size.height;
        float right_width = rightRect.size.width > rightRect.size.height
                                ? rightRect.size.width
                                : rightRect.size.height;
        float avage_length = (left_height + right_height) / 2;
        float avage_width = (left_width + right_width) / 2;
        // 计算装甲板对的最小外接矩形
        float min_x = static_cast<float>(
            std::min(leftRect.boundingRect().x, rightRect.boundingRect().x));
        float min_y = static_cast<float>(
            std::min(leftRect.boundingRect().y, rightRect.boundingRect().y));
        float max_x = static_cast<float>(std::max(
            leftRect.boundingRect().br().x, rightRect.boundingRect().br().x));
        float max_y = static_cast<float>(std::max(
            leftRect.boundingRect().br().y, rightRect.boundingRect().br().y));
        // 扩展装甲板对的最小外接矩形，用于确保完全包含数字
        float extension = static_cast<float>(avage_length * 0.5);
        min_y -= extension;
        max_y += extension;
        min_x += static_cast<float>(avage_width * 0.5);
        max_x -= static_cast<float>(avage_width * 0.5);
        // 取roi在图像范围内
        cv::Rect roi(cv::Point(std::max(0, static_cast<int>(min_x)),
                               std::max(0, static_cast<int>(min_y))),
                     cv::Point(std::min(image.cols, static_cast<int>(max_x)),
                               std::min(image.rows, static_cast<int>(max_y))));

        // 调整角度
        this->result = correctPerspective(image(roi), template_image);
        // 转换为灰度图
        cv::cvtColor(result, result, cv::COLOR_BGR2GRAY);
        if (!is_enemy_blue) {
          // 红色使用模板匹配函数
          if (!template_match(result,
                              0.32)) { // 0.5是相似度阈值，可以根据需要调整
            continue;                  // 跳过当前装甲板
          }
        }
        std::string result_str = classifier.classify_image(result);

        // 在矩形内绘制文字
        cv::Point2f rect_points[4];
        leftRect.points(rect_points);
        cv::Point textOrg(static_cast<int>(rect_points[0].x),
                          static_cast<int>(rect_points[0].y - 10));
        cv::putText(image, result_str, textOrg, cv::FONT_HERSHEY_SIMPLEX, 1.5,
                    cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
        cv::rectangle(image, roi, cv::Scalar(0, 255, 0), 2);
      }
    }
  }
};
class RobotDetector : public rclcpp::Node {
public:
  RobotDetector() : Node("robot_detector") {

    RCLCPP_INFO(this->get_logger(), "Hello, I am robot_detector node!");
    std::string model_path = "/home/ccy/workspace/deeplearning/model/"
                             "googlenet_100bitch_30_times9909.onnx";
    bool is_enemy_blue_;
    this->declare_parameter("is_enemy_blue", false);
    this->get_parameter("is_enemy_blue", is_enemy_blue_);
    dealor = std::make_unique<picture_dealor>(model_path, is_enemy_blue_);
    data_publisher_ =
        this->create_publisher<sensor_msgs::msg::Image>("debug_image", 10);
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(33),
        std::bind(&RobotDetector::timer_callback, this));
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "images_data", 10,
        std::bind(&RobotDetector::image_callback, this, std::placeholders::_1));
  }

private:
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr data_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::unique_ptr<picture_dealor> dealor;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  cv::Mat debug_result;
  void timer_callback() {
    std_msgs::msg::Header header;
    header.stamp = this->get_clock()->now();
    sensor_msgs::msg::Image::SharedPtr msg =
        cv_bridge::CvImage(header, "bgr8", this->debug_result).toImageMsg();
    data_publisher_->publish(*msg);
  }
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {

    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }
    cv::Mat img = cv_ptr->image;
    this->debug_result = dealor->color_detect(img);
  }
};
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RobotDetector>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

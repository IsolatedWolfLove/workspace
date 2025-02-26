#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

double distence_between_two_points(cv::Point2f &x, cv::Point2f &y) {
  return std::sqrt(std::pow(x.x - y.x, 2) + std::pow(x.y - y.y, 2));
}
class variance_calculator {
public:
  variance_calculator() {}
  double mean(const std::vector<float> &v) {
    return std::accumulate(v.begin(), v.end(), 0.0) /
           static_cast<double>(v.size());
  }

  double variance(const std::vector<float> &v) {
    double meanVal = mean(v);
    double sq_sum = std::accumulate(v.begin(), v.end(), 0.0,
                                    [meanVal](double acc, double val) {
                                      return acc + std::pow(val - meanVal, 2);
                                    });
    return sq_sum / static_cast<double>(v.size());
  }
};
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
  variance_calculator variance_cal;
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
      // image = 255 - image;
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
class picture_dealor {
  // param
  int binary_threshold = 75;
  float min_light_bar_area = 1000;
  float max_light_bar_area = 13000;
  float max_light_bar_width_length_diff = 1.2;
  float max_ligth_bar_angle = 30;
  float max_bin_point_y_diff = 0.3;
  int max_bin_length_diff = 30;
  double bin_diatance_times = 2.3;
  // param
public:
  picture_dealor(std::string &model_path_) : classifier(model_path_) {
    cv::Mat k;
    cv::Mat template_image =
        cv::imread("/home/ccy/workspace/finall_test/data/template_image.png",
                   cv::IMREAD_GRAYSCALE);
    this->template_image = template_image;
  }
  cv::Mat color_detect(cv::Mat &image) {
    this->image = image;
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    // bgr_subtract_and_blur
    cv::Mat gray, binary;
    if (is_enemy_blue) {
      cv::subtract(channels[0] + channels[1] * 0.4, channels[2], gray);
    } else {
      cv::subtract(channels[2] + channels[1] * 0.4, channels[0], gray);
    }
    // // 直方图均衡化
    // cv::equalizeHist(gray, gray);
    cv::imshow("gray", gray);
    cv::threshold(gray, binary, 75, 255, cv::THRESH_BINARY);
    cv::GaussianBlur(binary, binary, cv::Size(5, 5),
                     0); // 对蓝色通道进行高斯滤波

    // 进行闭操作
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(5, 5)); // 5x5 的矩形结构元素

    cv::Mat closed;
    cv::morphologyEx(binary, closed, cv::MORPH_CLOSE, kernel);

    return light_bar_detector(closed);
  }
  float angle_deal(cv::RotatedRect &rect) {
    float angle = rect.angle; // 计算矩形的角度
    if (rect.size.width > rect.size.height) {
      angle = 90 - angle; // 长边与竖直方向的夹角
    } else {
      angle = angle; // 直接使用 angle
    }
    return angle;
  }
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
  cv::Mat light_bar_detector(cv::Mat &image_closed) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image_closed, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE); // 寻找轮廓
    std::vector<cv::RotatedRect> rectangles; // 用于存储符合条件的矩形
    for (size_t i = 0; i < contours.size(); i++) {
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
      if ((area < min_light_bar_area) || (area > max_light_bar_area)) {
        continue;
      }
      if (std::abs(max_length - min_length) < max_light_bar_width_length_diff) {
        continue;
      }
      if ((std::abs(angle) > max_ligth_bar_angle)) {
        continue;
      }
      // for (int j = 0; j < 4; j++) {
      //   cv::line(image, rect_points[j], rect_points[(j + 1) % 4],
      //            cv::Scalar(0, 0, 255), 2); // 用红色绘制矩形
      // }
      rectangles.push_back(minRect);
    }
    return armour_detect(rectangles);
  }

  cv::Mat armour_detect(std::vector<cv::RotatedRect> &rectangles) {
    std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>> suitable_pairs_one;
    std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>> suitable_pairs_two;
    std::vector<std::pair<int, int>> suitable_index_one;

    for (size_t i = 0; i < rectangles.size(); i++) {
      for (size_t j = i + 1; j < rectangles.size(); j++) { // 对每对矩形进行比较
        float wide_i = rectangles[i].size.width; // 可使用 width 或 height
        float wide_j = rectangles[j].size.width;
        float height_i = rectangles[i].size.height; // 可使用 width 或 height
        float height_j = rectangles[j].size.height;
        float length_i = std::max(wide_i, height_i); // 长边
        float length_j = std::max(wide_j, height_j);
        float avage_length = (length_i + length_j) / 2; // 平均长度
        float angle_i = angle_deal(rectangles[i]);
        float angle_j = angle_deal(rectangles[j]);
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
        if ((std::abs(i_y - j_y) / avage_length) > max_bin_point_y_diff) {

          continue;
        }
        // 检查长度是否相近
        if ((std::pow(length_i - avage_length, 2) +
             std::pow(length_j - avage_length, 2)) < max_bin_length_diff) {
          // 计算中心点

          cv::Point2f center_i = rectangles[i].center;
          cv::Point2f center_j = rectangles[j].center;

          // 计算中心点之间的距离
          double distance = distence_between_two_points(center_i, center_j);
          double distance_threshold = bin_diatance_times * (avage_length);
          // 检查距离是否在阈值范围内
          if (distance < distance_threshold) {
            // 这里可以处理找到的两个相近的矩形
            // 可能在图像上绘制两条矩形连接线
            // cv::line(image, center_i, center_j, cv::Scalar(0, 255, 0),
            //          2); // 用绿色连接相近的矩形中心
            std::cout << (std::abs(i_y - j_y) / avage_length) << std::endl;
            suitable_pairs_one.push_back(
                std::make_pair(rectangles[i], rectangles[j]));
            suitable_index_one.push_back(std::make_pair(i, j));
          }
        }
      }
    }
    if (suitable_pairs_one.size() > 0) {
      for (auto &suits_pair : suitable_pairs_one) {
        cv::RotatedRect leftRect =
            suits_pair.first.center.x < suits_pair.second.center.x
                ? suits_pair.first
                : suits_pair.second;
        cv::RotatedRect rightRect =
            suits_pair.first.center.x > suits_pair.second.center.x
                ? suits_pair.first
                : suits_pair.second;
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
        // Find the bounding rectangle
        float min_x = static_cast<float>(
            std::min(leftRect.boundingRect().x, rightRect.boundingRect().x));
        float min_y = static_cast<float>(
            std::min(leftRect.boundingRect().y, rightRect.boundingRect().y));
        float max_x = static_cast<float>(std::max(
            leftRect.boundingRect().br().x, rightRect.boundingRect().br().x));
        float max_y = static_cast<float>(std::max(
            leftRect.boundingRect().br().y, rightRect.boundingRect().br().y));
        // Extend the width
        float extension = static_cast<float>(
            avage_length * 0.5); // Adjust this value as needed
        min_y -= extension;
        max_y += extension;
        min_x += static_cast<float>(avage_width * 0.5);
        max_x -= static_cast<float>(avage_width * 0.5);
        cv::Rect roi(cv::Point(std::max(0, static_cast<int>(min_x)),
                               std::max(0, static_cast<int>(min_y))),
                     cv::Point(std::min(image.cols, static_cast<int>(max_x)),
                               std::min(image.rows, static_cast<int>(max_y))));

        // Extract ROI
        // cv::imshow("roi", image(roi));
        this->result = correctPerspective(image(roi), template_image);
        cv::cvtColor(result, result, cv::COLOR_BGR2GRAY);
        // cv::imshow("result", result);
        std::string result_str = classifier.classify_image(result);
        // 找到第一个矩形的左上角位置
        cv::Point2f rect_points[4];
        leftRect.points(rect_points); // 获取第一个矩形的四个顶点
        cv::Point textOrg(static_cast<int>(rect_points[0].x),
                          static_cast<int>(rect_points[0].y -
                                           10)); // 左上角位置，向上移动10像素

        // 在矩形内绘制文字
        cv::putText(image, result_str, textOrg, cv::FONT_HERSHEY_SIMPLEX, 1.5,
                    cv::Scalar(255, 0, 255),
                    2, // Magenta color (B=255, G=0, R=255)
                    cv::LINE_AA);

        // Draw the bounding rectangle (optional, for visualization)
        cv::rectangle(image, roi, cv::Scalar(0, 255, 0), 2);
        return image;
      }
    }
    return image;
  }

private:
  bool is_enemy_blue = 1;
  cv::Mat image;
  cv::Mat template_image;
  cv::Mat result;
  ImageClassifier classifier; // 这里改为类的实例
};
int main() {
  std::string model_path = "/home/ccy/workspace/deeplearning/model/"
                           "googlenet_100bitch_30_times9909.onnx";
  cv::VideoCapture red_capture(
      "/home/ccy/workspace/finall_test/data/blue_armor_test.mp4");
  if (!red_capture.isOpened()) {
    std::cout << "red_capture open failed" << std::endl;
    return -1;
  }
  cv::Mat frame;
  cv::namedWindow("frame", cv::WINDOW_NORMAL);
  bool isPaused = false;
  int frame_count = 0;
  while (true) {
    // 如果不是暂停状态，读取新帧
    if (!isPaused) {
      red_capture >> frame;
    }

    // 检查是否成功读取到帧
    if (frame.empty()) {
      std::cout << "End of video or failed to read frame" << std::endl;
      break;
    }

    // 显示当前帧号
    std::string frameText = "Frame: " + std::to_string(frame_count);
    cv::putText(frame, frameText, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX,
                1, cv::Scalar(0, 255, 0), 2);

    cv::imshow("frame", frame);
    picture_dealor dealor(model_path);
    cv::Mat result = dealor.color_detect(frame);
    cv::imshow("result", result);

    char key = static_cast<char>(cv::waitKey(30));

    // 按 'p' 键切换暂停和继续
    if (key == 'p') {
      isPaused = !isPaused;
      std::cout << (isPaused ? "Video Paused" : "Video Playing") << std::endl;
    }

    // 暂停状态下的控制
    while (isPaused) {
      key = static_cast<char>(cv::waitKey(30));

      if (key == 'p') {
        isPaused = false;
        std::cout << "Video Playing" << std::endl;
        break;
      } else if (key == 'q') {
        goto cleanup; // 使用goto跳出多层循环
      } else if (key == 's') {
        std::string file_name = "/home/ccy/workspace/deeplearning/"
                                "learning_conda/src/load_model_cpp/loss/" +
                                std::to_string(frame_count) + ".png";
        cv::imwrite(file_name, frame);
        std::cout << "Saved: " << file_name << std::endl;
        frame_count++;
      } else if (key == 83) { // 右方向键
        for (int i = 0; i < 10; i++) {
          red_capture >> frame;
          if (frame.empty())
            break;
          frame_count++;
        }
        if (!frame.empty()) {
          cv::imshow("frame", frame);
          picture_dealor dealor(model_path);
          cv::Mat result = dealor.color_detect(frame);
          cv::imshow("result", result);
        }
      } else if (key == 81) { // 左方向键
        int target_pos = std::max(
            0, static_cast<int>(red_capture.get(cv::CAP_PROP_POS_FRAMES)) - 5);
        red_capture.set(cv::CAP_PROP_POS_FRAMES, target_pos);
        red_capture >> frame;
        if (!frame.empty()) {
          cv::imshow("frame", frame);
          picture_dealor dealor(model_path);
          cv::Mat result = dealor.color_detect(frame);
          cv::imshow("result", result);
          frame_count = target_pos;
        }
      }
    }

    // 非暂停状态下的方向键控制
    if (key == 83) { // 右方向键
      for (int i = 0; i < 10; i++) {
        red_capture >> frame;
        if (frame.empty())
          break;
        frame_count++;
      }
      if (!frame.empty()) {
        cv::imshow("frame", frame);
        picture_dealor dealor(model_path);
        cv::Mat result = dealor.color_detect(frame);
        cv::imshow("result", result);
      }
    } else if (key == 81) { // 左方向键
      int target_pos = std::max(
          0, static_cast<int>(red_capture.get(cv::CAP_PROP_POS_FRAMES)) - 5);
      red_capture.set(cv::CAP_PROP_POS_FRAMES, target_pos);
      red_capture >> frame;
      if (!frame.empty()) {
        cv::imshow("frame", frame);
        picture_dealor dealor(model_path);
        cv::Mat result = dealor.color_detect(frame);
        cv::imshow("result", result);
        frame_count = target_pos;
      }
    }

    // 按 'q' 键退出
    if (key == 'q') {
      break;
    }

    // 如果不是暂停状态，更新帧计数
    if (!isPaused) {
      frame_count++;
    }
  }

cleanup:
  red_capture.release();
  cv::destroyAllWindows();
  return 0;
}

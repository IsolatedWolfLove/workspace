#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <thread>
#include <vector>
int main() {
  std::array<float, 2> base_x;
  std::array<float, 2> base_y;
  std::array<float, 2> target_x;
  std::array<float, 2> target_y;
  std::vector<float> target_base_x;
  std::vector<float> target_base_y;
  std::vector<float> base_world_x;
  std::vector<float> base_world_y;
  float v = 20;
  std::fstream file;
  file.open("data1.txt", std::ios::in);
  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      std::string sub_line1 = line.substr(0, line.find("  "));
      std::string sub_line2 = line.substr(line.find("  ") + 2,
                                          line.find("  ", line.find("  ") + 1));
      std::string sub_line3 = line.substr(
          line.find("  ", line.find("  ") + 1) + 2,
          line.find("  ", line.find("  ", line.find("  ") + 1) + 1));
      std::string sub_line4 = line.substr(
          line.find("  ", line.find("  ", line.find("  ") + 1) + 1) + 2);

      target_base_x.push_back(std::stod(sub_line1));
      target_base_y.push_back(std::stod(sub_line2));
      base_world_x.push_back(std::stod(sub_line3));
      base_world_y.push_back(std::stod(sub_line4));
    }
    file.close();
  } else {
    std::cout << "Error: unable to open file";
  }

  float time_step = 0.02;
    float correct_x = 0.0;
    float correct_y = 0.0;
    for (int i = 0; i < target_base_x.size(); i += 2) {
      base_x[0] = base_world_x[i];
      base_y[0] = base_world_y[i];
      target_x[0] = target_base_x[i] + base_world_x[i];
      target_y[0] = target_base_y[i] + base_world_y[i];
      base_x[1] = base_world_x[i + 1];
      base_y[1] = base_world_y[i + 1];
      target_x[1] = target_base_x[i + 1] + base_world_x[i + 1];
      target_y[1] = target_base_y[i + 1] + base_world_y[i + 1];
      float distance = sqrt(pow(target_x[1] - base_x[1], 2) +
                            pow(target_y[1] - base_y[1], 2));
      std::cout << "distance: " << distance << std::endl;
      float time = distance / v;
      float p_n_x = (target_x[1] - base_x[0]) > 0 ? 1 : -1;
      float p_n_y = (target_y[1] - base_y[0]) > 0 ? 1 : -1;
      float arm_v = sqrt(pow(target_y[1] - target_y[0], 2) +
                         pow(target_x[1] - target_x[0], 2)) /
                    time_step;
      float pre_s = arm_v * time;
      float tan_theta_squ = std::pow(
          (target_y[1] - target_y[0]) / (target_x[1] - target_x[0]), 2);
      float cos_theta = std::sqrt(1.0 / (tan_theta_squ + 1.0));
      float sin_theta = std::sqrt(tan_theta_squ / (tan_theta_squ + 1.0));
      float pre_target_x = target_x[1] + p_n_x * pre_s * cos_theta + correct_x;
      float pre_target_y = target_y[1] + p_n_y * pre_s * sin_theta + correct_y;
      float pre_tan_theta =
          (pre_target_y - base_y[1]) / (pre_target_x - base_x[1]);
      float pre_theta = std::atan(pre_tan_theta);
      float target_3_x = target_base_x[i + 2] + base_world_x[i + 2];
      float target_3_y = target_base_y[i + 2] + base_world_y[i + 2];
      if (sqrt(pow(pre_target_x - target_3_x, 2) +
               pow(pre_target_y - target_3_y, 2)) < 0.25) {
        std::cout << "reach target 3,fire!!!" << std::endl;
      } else {
        std::cout << "sorry,not reach target 3,try again!!!" << std::endl;
        correct_x = -(pre_target_x - target_3_x)/2.68;
        correct_y = -(pre_target_y - target_3_y)/2.68;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

  return 0;
}

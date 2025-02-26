#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

void drawFood(cv::Mat &drawingBuffer, const cv::Point &food, int is_big) {
  if (is_big == 5) {
    cv::circle(drawingBuffer, food, 30, cv::Scalar(255, 255, 255), -1);
  } else {
    cv::circle(drawingBuffer, food, 10, cv::Scalar(255, 255, 255), -1);
  }
}

void drawSnake(cv::Mat &drawingBuffer, const std::vector<cv::Point> &snake) {
  // 绘制蛇头
  cv::circle(drawingBuffer, snake[0], 10, cv::Scalar(0, 0, 255), -1);
  // 绘制蛇身
  for (size_t i = 1; i < snake.size(); i++) {
    cv::circle(drawingBuffer, snake[i], 10, cv::Scalar(0, 255, 255), -1);
  }
}

int main() {
  int speed = 100; // 游戏速度
  int level = 1;   // 难度等级
  std::cout << "请输入难度等级(1-5):";
  std::cin >> level;
  speed = 100 - level * 10; // 根据难度等级调整速度
  // 初始化游戏环境
  int width = 1200, height = 1200;
  int lifes = 3;  // 生命数
  int is_big = 0; // 是否变大
  cv::Mat drawingBuffer(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat displayBuffer(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::namedWindow("Snake Game");

  // 游戏状态和变量
  std::vector<cv::Point> snake;
  snake.push_back(cv::Point(width / 2, height / 2));

  char direction = 'L'; // 初始方向为右
  bool gameRunning = true;

  // 创建一个随机数引擎，使用默认的随机数种子
  std::random_device rd;  // 非确定性随机数生成器
  std::mt19937 gen(rd()); // 使用Mersenne Twister算法的随机数生成器
  std::uniform_int_distribution<> distrib(0, width / 10 - 1);

  cv::Point food(distrib(gen) * 10, distrib(gen) * 10); // 随机生成食物
  int score = 0;                                        // 分数
  // 游戏循环
  while (gameRunning) {
    // 绘制当前帧
    drawingBuffer.setTo(cv::Scalar(0, 0, 0)); // 清空绘制缓冲区
    drawSnake(drawingBuffer, snake);
    drawFood(drawingBuffer, food, is_big);
    std::string scoreText =
        "Score: " + std::to_string(score) + "\nLifes: " + std::to_string(lifes);
    cv::putText(drawingBuffer, scoreText, cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    // 处理用户输入
    int key = cv::waitKey(speed);
    if (key == 27) { // 按下ESC键退出游戏
      gameRunning = false;
    } else if (key == 82) { // 上键
      direction = 'U';
    } else if (key == 84) { // 下键
      direction = 'D';
    } else if (key == 81) { // 左键
      direction = 'L';
    } else if (key == 83) { // 右键
      direction = 'R';
    }

    // 移动蛇头
    cv::Point newHead = snake[0];
    switch (direction) {
    case 'U':
      newHead.y -= 20;
      break;
    case 'D':
      newHead.y += 20;
      break;
    case 'L':
      newHead.x -= 20;
      break;
    case 'R':
      newHead.x += 20;
      break;
    }

    // 更新蛇的位置
    snake.insert(snake.begin(), newHead);
    snake.pop_back(); // 移除蛇尾

    // 碰撞检测
    if (newHead.x < 0 || newHead.x >= width || newHead.y < 0 ||
        newHead.y >= height) {
      lifes--;          // 生命数减少
      if (lifes == 0) { // 生命数为0，结束游戏
        gameRunning = false;
      }
      snake.clear();                                     // 清空蛇
      snake.push_back(cv::Point(width / 2, height / 2)); // 重置蛇头
      direction = 'R';                                   // 重置方向
    }
    for (size_t i = 1; i < snake.size(); i++) {
      if (snake[0] == snake[i]) {
        lifes--;               // 生命数减少
        if (lifes == 0) {      // 生命数为0，结束游戏
          gameRunning = false; // 碰到自己，结束游戏
        }
        snake.clear();                                     // 清空蛇
        snake.push_back(cv::Point(width / 2, height / 2)); // 重置蛇头
        direction = 'R';                                   // 重置方向
      }
    }
    int thead = (is_big == 5 ? 70 : 25);
    // 吃到食物
    if (abs(snake[0].x - food.x) + abs(snake[0].y - food.y) < thead) {
      snake.push_back(snake.back()); // 增加一节蛇身
      food = cv::Point(distrib(gen) * 10, distrib(gen) * 10);
      if (is_big == 5) {
        score += 50;
        is_big = 0;

      } else {
        score += 10;
        is_big++;
      } // 生成新的食物
    }

    // 交换缓冲区
    cv::Mat temp = drawingBuffer.clone();
    displayBuffer = temp;

    // 显示内容
    cv::imshow("Snake Game", displayBuffer);
  }

  // 游戏结束
  cv::destroyAllWindows();
  return 0;
}
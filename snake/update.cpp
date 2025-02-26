#include <ctime>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#define PI 3.14159265358979323846

const std::string GAME_INSTRUCTIONS = R"(
=== 贪吃蛇游戏说明 ===
1. 控制: 方向键控制移动
2. 食物:
   - 白色: 普通食物 (+10分)
   - 绿色: 加速食物 (+20分, 加速)
   - 红色: 减速食物 (+5分, 减速)
   - 蓝色: 生命食物 (+15分, +1生命)
3. 道具 (五角星):
   - 金色: 无敌 (15秒)
   - 青色: 穿墙 (15秒)
   - 紫色: 双倍分数 (15秒)
4. 敌人:
   - 红色弓箭形状
   - 每20秒生成一个
   - 按q发射子弹消灭敌人 (-30分)
5. 游戏结束:
   - 生命耗尽
   - 被敌人抓住
   在开始之前，我希望给用户显示一个游戏说明：各个道具是啥功能之类的；我还希望把道具的形状改成五角星来和食物分开，同时道具可以和食物共存，屏幕上最多可以有一个食物和一个道具；再增加一个玩法：每隔20s生成一个敌人（恒定在左上角生成，颜色为红色，形状类似一个弓箭），敌人会追逐玩家，模仿玩家动作，且在玩家沿直线行驶时，敌人速度为玩家的两倍，在玩家转弯时，敌人转弯花费时间比玩家多3倍，玩家可以通过消耗30分来发出一个子弹（颜色为红色，形状为三角形）按q键发出，子弹只能直线行驶，当碰到敌人（只需检验在敌人附近）就会爆炸杀死敌人，注意：当玩家在二十秒内没有杀死敌人，这个敌人不会消失，但会有一个新敌人生成，即此时有两个敌人追逐玩家，当敌人咬住了玩家任意部位，全场上所有敌人均消失，玩家生命数减去战场上敌人总数  @/update.cpp
)";

struct PowerUp {
  cv::Point position;
  int type;
  time_t spawnTime;
};

struct Enemy {
  cv::Point position;
  cv::Point velocity;
  time_t spawnTime;
  bool active;
};

struct Bullet {
  cv::Point position;
  cv::Point direction;
  bool active;
};

void drawFood(cv::Mat &drawingBuffer, const cv::Point &food, int type) {
  if (type == 1) {
    cv::circle(drawingBuffer, food, 10, cv::Scalar(255, 255, 255), -1);
  } else if (type == 2) {
    cv::circle(drawingBuffer, food, 10, cv::Scalar(0, 255, 0), -1);
  } else if (type == 3) {
    cv::circle(drawingBuffer, food, 10, cv::Scalar(0, 0, 255), -1);
  } else if (type == 4) {
    cv::circle(drawingBuffer, food, 10, cv::Scalar(255, 0, 0), -1);
  }
}

void drawSnake(cv::Mat &drawingBuffer, const std::vector<cv::Point> &snake) {
  cv::circle(drawingBuffer, snake[0], 10, cv::Scalar(0, 0, 255), -1);
  for (size_t i = 1; i < snake.size(); i++) {
    cv::circle(drawingBuffer, snake[i], 10, cv::Scalar(0, 255, 255), -1);
  }
}

void drawObstacles(cv::Mat &drawingBuffer,
                   const std::vector<cv::Point> &obstacles) {
  for (const auto &obstacle : obstacles) {
    cv::rectangle(drawingBuffer, cv::Rect(obstacle.x, obstacle.y, 20, 20),
                  cv::Scalar(128, 128, 128), -1);
  }
}

// 绘制五角星
void drawStar(cv::Mat &img, cv::Point center, int radius, cv::Scalar color) {
  std::vector<cv::Point> points;

  for (int i = 0; i < 5; i++) {
    double angle = i * 4 * PI / 5 - PI / 2;
    points.push_back(cv::Point(center.x + radius * cos(angle),
                               center.y + radius * sin(angle)));
    angle += 2 * PI / 5;
    points.push_back(cv::Point(center.x + radius / 2 * cos(angle),
                               center.y + radius / 2 * sin(angle)));
  }

  cv::fillConvexPoly(img, points, color);
}

// 绘制弓箭形状的敌人
void drawEnemy(cv::Mat &img, cv::Point center, cv::Point velocity) {
  // 绘制弓身
  cv::line(img, cv::Point(center.x - 15, center.y - 5),
           cv::Point(center.x + 15, center.y - 5), cv::Scalar(0, 0, 255), 2);

  // 绘制弓弦
  cv::line(img, cv::Point(center.x - 15, center.y - 5),
           cv::Point(center.x + 15, center.y - 5), cv::Scalar(0, 0, 255), 2);

  // 绘制箭头
  cv::Point arrowTip = center + velocity * 20;
  cv::line(img, center, arrowTip, cv::Scalar(0, 0, 255), 2);
  cv::circle(img, arrowTip, 3, cv::Scalar(0, 0, 255), -1);
}

// 绘制子弹
void drawBullet(cv::Mat &img, cv::Point position) {
  std::vector<cv::Point> triangle = {
      cv::Point(position.x - 5, position.y + 10),
      cv::Point(position.x + 5, position.y + 10)};
  cv::fillConvexPoly(img, triangle, cv::Scalar(0, 0, 255));
}

void drawPowerUps(cv::Mat &drawingBuffer,
                  const std::vector<PowerUp> &powerUps) {
  for (const auto &powerUp : powerUps) {
    cv::Scalar color;
    switch (powerUp.type) {
    case 0:
      color = cv::Scalar(255, 215, 0);
      break; // 无敌 - 金色
    case 1:
      color = cv::Scalar(0, 255, 255);
      break; // 穿墙 - 青色
    case 2:
      color = cv::Scalar(255, 0, 255);
      break; // 双倍分数 - 紫色
    }
    drawStar(drawingBuffer, powerUp.position, 15, color);
    cv::putText(drawingBuffer, std::to_string(powerUp.type),
                cv::Point(powerUp.position.x - 10, powerUp.position.y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
  }
}

int main() {
  int speed = 100; // 游戏速度
  int level = 1;   // 难度等级
  std::cout << "请输入难度等级(1-5):";
  std::cin >> level;
  speed = 100 - level * 10; // 根据难度等级调整速度
  // 显示游戏说明并等待用户输入
  speed = 100 - level * 10;
  cv::Mat instructionBuffer(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::putText(instructionBuffer, GAME_INSTRUCTIONS, cv::Point(50, 50),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
  cv::putText(instructionBuffer, "按任意键开始游戏", cv::Point(50, height - 50),
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
  cv::imshow("Snake Game", instructionBuffer);
  cv::waitKey(0);

  // 初始化游戏环境
  int width = 1200, height = 1200;
  int lifes = 5; // 生命数
  cv::Mat drawingBuffer(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat displayBuffer(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::namedWindow("Snake Game");
  time_t gameStartTime = time(nullptr); // 游戏开始时间

  // 游戏状态和变量
  std::vector<cv::Point> snake;
  snake.push_back(cv::Point(width / 2, height / 2));

  char direction = 'L'; // 初始方向为右
  bool gameRunning = true;

  // 敌人系统
  std::vector<Enemy> enemies;
  time_t lastEnemySpawn = time(nullptr);
  int maxEnemies = 2; // 最多同时存在2个敌人

  // 子弹系统
  std::vector<Bullet> bullets;
  int bulletSpeed = 10; // 子弹速度

  // 特殊道具状态
  std::vector<PowerUp> powerUps;
  bool invincible = false;  // 无敌状态
  bool wallPass = false;    // 穿墙状态
  bool doubleScore = false; // 双倍分数状态

  // 创建一个随机数引擎，使用默认的随机数种子
  std::random_device rd;  // 非确定性随机数生成器
  std::mt19937 gen(rd()); // 使用Mersenne Twister算法的随机数生成器
  std::uniform_int_distribution<> distrib(0, width / 10 - 1);

  cv::Point food(distrib(gen) * 10, distrib(gen) * 10); // 随机生成食物
  int foodType = 1;                                     // 食物类型
  int score = 0;                                        // 分数

  // 关卡系统
  int currentLevel = 1;
  int maxLevel = 5;
  std::vector<cv::Point> obstacles;
  auto generateObstacles = [&]() {
    obstacles.clear();
    int obstacleCount = 20 + (currentLevel - 1) * 5; // 每关增加5个障碍物
    int obstacleCount = 20 + (currentLevel - 1) * 5;
    for (int i = 0; i < obstacleCount; ++i) {
      cv::Point newObstacle(distrib(gen) * 10, distrib(gen) * 10);
      // 确保障碍物不与蛇初始位置重叠
      while (abs(newObstacle.x - width / 2) < 100 &&
             abs(newObstacle.y - height / 2) < 100) {
        newObstacle = cv::Point(distrib(gen) * 10, distrib(gen) * 10);
      }
      obstacles.push_back(newObstacle);
    }
  };
  generateObstacles();

  // 道具生成函数
  auto spawnPowerUp = [&]() {
        if (powerUps.size() < 2) { // 最多同时存在2个道具
        if (powerUps.size() < 2) {
            PowerUp newPowerUp;
            newPowerUp.position = cv::Point(distrib(gen) * 10, distrib(gen) * 10);
            newPowerUp.type = distrib(gen) % 3;
            newPowerUp.spawnTime = time(nullptr);
            powerUps.push_back(newPowerUp);
        }
    };

    // 道具拾取检测
    auto checkPowerUpCollision = [&]() {
        for (auto it = powerUps.begin(); it != powerUps.end(); ) {
            if (abs(snake[0].x - it->position.x) + abs(snake[0].y - it->position.y) < 25) {
                switch (it->type) {
                    case 0: // 无敌
                    case 0:
                        invincible = true;
                        break;
                    case 1: // 穿墙
                    case 1:
                        wallPass = true;
                        break;
                    case 2: // 双倍分数
                    case 2:
                        doubleScore = true;
                        break;
                }
                it = powerUps.erase(it);
            } else {
                ++it;
            }
        }
    };

    time_t foodSpawnTime = time(nullptr); // 食物生成时间
    time_t foodSpawnTime = time(nullptr);

    // 游戏循环
    while (gameRunning) {
        // 绘制当前帧
        drawingBuffer.setTo(cv::Scalar(0, 0, 0)); // 清空绘制缓冲区
        drawingBuffer.setTo(cv::Scalar(0, 0, 0));
        drawSnake(drawingBuffer, snake);
        drawFood(drawingBuffer, food, foodType);
        drawObstacles(drawingBuffer, obstacles);
        drawPowerUps(drawingBuffer, powerUps);

        // 每10秒生成一个道具
        static time_t lastPowerUpSpawn = time(nullptr);
        if (time(nullptr) - lastPowerUpSpawn > 10) {
            spawnPowerUp();
            lastPowerUpSpawn = time(nullptr);
        }

        // 检查并移除过期道具
        powerUps.erase(std::remove_if(powerUps.begin(), powerUps.end(),
            [](const PowerUp& p) {
                return time(nullptr) - p.spawnTime > 10;
            }), powerUps.end());

        // 检查道具拾取
        checkPowerUpCollision();

        // 更新道具状态
        static time_t powerUpStartTime = 0;
        if (invincible || wallPass || doubleScore) {
            if (powerUpStartTime == 0) {
                powerUpStartTime = time(nullptr);
            }
            if (time(nullptr) - powerUpStartTime > 15) { // 道具效果持续15秒
            if (time(nullptr) - powerUpStartTime > 15) {
                invincible = false;
                wallPass = false;
                doubleScore = false;
                powerUpStartTime = 0;
            }
        }

        // 更新敌人位置
        for (auto& enemy : enemies) {
            if (!enemy.active) continue;
        // 每20秒生成一个敌人，最多同时存在maxEnemies个
        if (time(nullptr) - lastEnemySpawn > 20 && enemies.size() < maxEnemies) {
            Enemy newEnemy;
            newEnemy.position = cv::Point(0, 0); // 在左上角生成
            newEnemy.velocity = cv::Point(1, 1); // 初始速度
            newEnemy.spawnTime = time(nullptr);
            newEnemy.active = true;
            enemies.push_back(newEnemy);
            lastEnemySpawn = time(nullptr);
        }
            
            // 计算敌人移动方向
            cv::Point target = snake[0] - enemy.position;
            double dist = cv::norm(target);
            if (dist > 0) {
                target = target * (1.0 / dist); // 归一化

                // 敌人速度是玩家的2倍（直线）或1/3（转弯）
                if (direction == 'U' || direction == 'D' || direction == 'L' || direction == 'R') {
                    enemy.velocity = target * 2.0;
                } else {
                    enemy.velocity = target * 0.33;
                }

                // 更新敌人位置
                enemy.position += enemy.velocity;

                // 检查敌人是否抓住玩家
                if (cv::norm(enemy.position - snake[0]) < 20) {
                    lifes -= enemies.size();
                    if (lifes <= 0) {
                        gameRunning = false;
                    }
                    enemies.clear(); // 清除所有敌人
                }
            }

            // 绘制敌人
            drawEnemy(drawingBuffer, enemy.position, enemy.velocity);
        }

        // 更新子弹位置
        for (auto& bullet : bullets) {
            if (!bullet.active) continue;

            bullet.position += bullet.direction * bulletSpeed;

            // 检查子弹是否出界
            if (bullet.position.x < 0 || bullet.position.x >= width ||
                bullet.position.y < 0 || bullet.position.y >= height) {
                bullet.active = false;
                continue;
            }

            // 绘制子弹
            drawBullet(drawingBuffer, bullet.position);

            // 检查子弹与敌人的碰撞
            for (auto& enemy : enemies) {
                if (enemy.active && cv::norm(bullet.position - enemy.position) < 20) {
                bullet.active = false;
                enemy.active = false;
                score += 50; // 消灭敌人奖励50分
                break;
            if (!enemy.active) continue;
            
            cv::Point target = snake[0] - enemy.position;
            double dist = cv::norm(target);
            if (dist > 0) {
                target = target * (1.0 / dist);
                
                if (direction == 'U' || direction == 'D' || direction == 'L' || direction == 'R') {
                    enemy.velocity = target * 2.0;
                } else {
                    enemy.velocity = target * 0.33;
                }
                
                enemy.position += enemy.velocity;
            }
            
            drawEnemy(drawingBuffer, enemy.position, enemy.velocity);
        }
        
        // 移除失效的子弹和敌人
        bullets.erase(std::remove_if(bullets.begin(), bullets.end(),
            [](const Bullet& b) { return !b.active; }), bullets.end());

        enemies.erase(std::remove_if(enemies.begin(), enemies.end(),
            [](const Enemy& e) { return !e.active; }), enemies.end());

        // 更新分数显示
        std::string scoreText = "Score: " + std::to_string(score);
        if (doubleScore) {
            scoreText += " (2x)";
        }
        scoreText += "\nLifes: " + std::to_string(lifes);
        if (invincible) {
            scoreText += " (Invincible)";
        }
        if (wallPass) {
            scoreText += " (Wall Pass)";
        }
        cv::putText(drawingBuffer, scoreText, cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        // 添加道具效果可视化
        if (invincible) {
            cv::circle(drawingBuffer, snake[0], 15, cv::Scalar(255, 215, 0), 3); // 金色光环
        }
        if (wallPass) {
            cv::circle(drawingBuffer, snake[0], 18, cv::Scalar(0, 255, 255), 3); // 青色光环
        }
        if (doubleScore) {
            cv::circle(drawingBuffer, snake[0], 21, cv::Scalar(255, 0, 255), 3); // 紫色光环
        }
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
        } else if (key == 'q' && score >= 30) { // 发射子弹
            Bullet newBullet;
            newBullet.position = snake[0];
            switch (direction) {
                case 'U': newBullet.direction = cv::Point(0, -1); break;
                case 'D': newBullet.direction = cv::Point(0, 1); break;
                case 'L': newBullet.direction = cv::Point(-1, 0); break;
                case 'R': newBullet.direction = cv::Point(1, 0); break;
            }
            newBullet.active = true;
            bullets.push_back(newBullet);
            score -= 30; // 消耗30分
        }

        // 移动蛇头
        cv::Point newHead = snake[0];
        switch (direction) {
        case 'U':
            newHead.y -= 20;
        for (auto& bullet : bullets) {
            if (!bullet.active) continue;
            
            bullet.position += bullet.direction * 10;
            
            if (bullet.position.x < 0 || bullet.position.x >= width ||
                bullet.position.y < 0 || bullet.position.y >= height) {
                bullet.active = false;
                continue;
            }
            
            drawBullet(drawingBuffer, bullet.position);
            
            for (auto& enemy : enemies) {
                if (enemy.active && cv::norm(bullet.position - enemy.position) < 20) {
                    bullet.active = false;
                    enemy.active = false;
                    score += 50;
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

        // 碰撞检测（考虑无敌和穿墙效果）
        if (!invincible) {
            // 墙壁碰撞检测（穿墙时忽略）
            if (!wallPass && (newHead.x < 0 || newHead.x >= width || newHead.y < 0 || newHead.y >= height)) {
                lifes--;          // 生命数减少
                if (lifes == 0) { // 生命数为0，结束游戏
                    gameRunning = false;
                }
                snake.clear();                                     // 清空蛇
                snake.push_back(cv::Point(width / 2, height / 2)); // 重置蛇头
                direction = 'R';                                   // 重置方向
            }

            // 自身碰撞检测
            for (size_t i = 1; i < snake.size(); i++) {
                if (snake[0] == snake[i]) {
                    lifes--;               // 生命数减少
                    if (lifes == 0) {      // 生命数为0，结束游戏
                        gameRunning = false; // 碰到自己，结束游戏
                    }
                    snake.clear();                                     // 清空蛇
                    snake.push_back(cv::Point(width / 2, height / 2)); // 重置蛇头
                    direction = 'R';                                   // 重置方向
                    break;
                }
            }

            // 障碍物碰撞检测
            for (const auto &obstacle : obstacles) {
                if (abs(snake[0].x - obstacle.x) < 20 && abs(snake[0].y - obstacle.y) < 20) {
                    lifes--;
                    if (lifes == 0) {
                        gameRunning = false;
                    }
                    snake.clear();
                    snake.push_back(cv::Point(width / 2, height / 2));
                    direction = 'R';
                    break;
                }
            }
        }

        // 穿墙效果处理
                }
            }
        }
        
        bullets.erase(std::remove_if(bullets.begin(), bullets.end(),
            [](const Bullet& b) { return !b.active; }), bullets.end());
            
        enemies.erase(std::remove_if(enemies.begin(), enemies.end(),
            [](const Enemy& e) { return !e.active; }), enemies.end());
            
        std::string scoreText = "Score: " + std::to_string(score);
        if (doubleScore) {
            scoreText += " (2x)";
        }
        scoreText += "\nLifes: " + std::to_string(lifes);
        if (invincible) {
            scoreText += " (Invincible)";
        }
        if (wallPass) {
            if (newHead.x < 0) newHead.x = width - 20;
            else if (newHead.x >= width) newHead.x = 0;
            if (newHead.y < 0) newHead.y = height - 20;
            else if (newHead.y >= height) newHead.y = 0;
            scoreText += " (Wall Pass)";
        }
    } // 添加缺失的闭合大括号
        cv::putText(drawingBuffer, scoreText, cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        // 吃到食物
        if (abs(snake[0].x - food.x) + abs(snake[0].y - food.y) < 25) {
            snake.push_back(snake.back()); // 增加一节蛇身
            food = cv::Point(distrib(gen) * 10, distrib(gen) * 10);
            foodType = distrib(gen) % 4 + 1; // 随机生成食物类型
            foodSpawnTime = time(nullptr); // 重置食物生成时间

            int scoreMultiplier = doubleScore ? 2 : 1;
            switch (foodType) {
            case 1: // 普通食物
                score += 10 * scoreMultiplier;
        if (invincible) {
            cv::circle(drawingBuffer, snake[0], 15, cv::Scalar(255, 215, 0), 3);
        }
        if (wallPass) {
            cv::circle(drawingBuffer, snake[0], 18, cv::Scalar(0, 255, 255), 3);
        }
        if (doubleScore) {
            cv::circle(drawingBuffer, snake[0], 21, cv::Scalar(255, 0, 255), 3);
        }

        int key = cv::waitKey(speed);
        if (key == 27) {
            gameRunning = false;
        } else if (key == 82) {
            direction = 'U';
        } else if (key == 84) {
            direction = 'D';
        } else if (key == 81) {
            direction = 'L';
        } else if (key == 83) {
            direction = 'R';
        } else if (key == 'q' && score >= 30) {
            Bullet newBullet;
            newBullet.position = snake[0];
            switch (direction) {
                case 'U': newBullet.direction = cv::Point(0, -1); break;
                case 'D': newBullet.direction = cv::Point(0, 1); break;
                case 'L': newBullet.direction = cv::Point(-1, 0); break;
                case 'R': newBullet.direction = cv::Point(1, 0); break;
            }
            newBullet.active = true;
            bullets.push_back(newBullet);
            score -= 30;
        }

        cv::Point newHead = snake[0];
        switch (direction) {
        case 'U':
            newHead.y -= 20;
                break;
            case 2: // 加速食物
                speed = std::max(50, speed - 20);
                score += 20 * scoreMultiplier;
        case 'D':
            newHead.y += 20;
                break;
            case 3: // 减速食物
                speed = std::min(200, speed + 20);
                score += 5 * scoreMultiplier;
        case 'L':
            newHead.x -= 20;
                break;
            case 4: // 生命值增加食物
                lifes++;
                score += 15 * scoreMultiplier;
                break;
        case 'R':
            newHead.x += 20;
            break;
            }

        snake.insert(snake.begin(), newHead);
        snake.pop_back();

        if (!invincible) {
            if (!wallPass && (newHead.x < 0 || newHead.x >= width || newHead.y < 0 || newHead.y >= height)) {
                lifes--;
                if (lifes == 0) {
                    gameRunning = false;
        }
                snake.clear();
                snake.push_back(cv::Point(width / 2, height / 2));
                direction = 'R';
            }

        // 食物消失并生成新食物
        if (time(nullptr) - foodSpawnTime > 5) { // 食物存在超过5秒
            for (size_t i = 1; i < snake.size(); i++) {
                if (snake[0] == snake[i]) {
                    lifes--;
                    if (lifes == 0) {
                        gameRunning = false;
                    }
                    snake.clear();
                    snake.push_back(cv::Point(width / 2, height / 2));
                    direction = 'R';
                    break;
                }
            }

            for (const auto &obstacle : obstacles) {
                if (abs(snake[0].x - obstacle.x) < 20 && abs(snake[0].y - obstacle.y) < 20) {
                    lifes--;
                    if (lifes == 0) {
                        gameRunning = false;
                    }
                    snake.clear();
                    snake.push_back(cv::Point(width / 2, height / 2));
                    direction = 'R';
                    break;
                }
            }
        }

        if (wallPass) {
            if (newHead.x < 0) newHead.x = width - 20;
            else if (newHead.x >= width) newHead.x = 0;
            if (newHead.y < 0) newHead.y = height - 20;
            else if (newHead.y >= height) newHead.y = 0;
        }

        if (abs(snake[0].x - food.x) + abs(snake[0].y - food.y) < 25) {
            snake.push_back(snake.back());
            food = cv::Point(distrib(gen) * 10, distrib(gen) * 10);
            foodType = distrib(gen) % 4 + 1;
            foodSpawnTime = time(nullptr);
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

                if (abs(snake[0].x - obstacle.x) < 20 &&


            int scoreMultiplier = doubleScore ? 2 : 1;
            switch (foodType) {
            case 1:
                score += 10 * scoreMultiplier;
                break;
            case

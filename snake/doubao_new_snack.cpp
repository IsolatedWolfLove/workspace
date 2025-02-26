#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>

const int WIDTH = 800;
const int HEIGHT = 600;
const int SNAKE_SIZE = 20;
const int FOOD_SIZE = 20;
const int PROP_SIZE = 20;
const int ENEMY_SIZE = 20;
const int BULLET_SIZE = 10;

class Point {
public:
    int x;
    int y;
    Point(int _x, int _y) : x(_x), y(_y) {}
};

class Snake {
public:
    std::vector<Point> body;
    int direction; // 0: up, 1: right, 2: down, 3: left
    int speed;
    int score;
    int lives;

    Snake() {
        body.emplace_back(WIDTH / 2, HEIGHT / 2);
        direction = 1;
        speed = 10;
        score = 0;
        lives = 3;
    }

    void move() {
        Point head = body[0];
        switch (direction) {
            case 0: head.y -= speed; break;
            case 1: head.x += speed; break;
            case 2: head.y += speed; break;
            case 3: head.x -= speed; break;
        }
        body.insert(body.begin(), head);
        body.pop_back();
    }

    bool checkSelfCollision() {
        for (size_t i = 1; i < body.size(); ++i) {
            if (body[0].x == body[i].x && body[0].y == body[i].y) {
                return true;
            }
        }
        return false;
    }
};

class Food {
public:
    Point position;
    int type; // 0: white, 1: green, 2: red, 3: blue
    Food() {
        generatePosition();
        type = std::rand() % 4;
    }

    void generatePosition() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> disX(0, (WIDTH - FOOD_SIZE) / SNAKE_SIZE);
        std::uniform_int_distribution<> disY(0, (HEIGHT - FOOD_SIZE) / SNAKE_SIZE);
        position.x = disX(gen) * SNAKE_SIZE;
        position.y = disY(gen) * SNAKE_SIZE;
    }
};

class Prop {
public:
    Point position;
    int type; // 0: golden, 1: cyan, 2: purple
    bool active;
    int duration;
    Prop() : active(false), duration(0) {
        generatePosition();
        type = std::rand() % 3;
    }

    void generatePosition() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> disX(0, (WIDTH - PROP_SIZE) / SNAKE_SIZE);
        std::uniform_int_distribution<> disY(0, (HEIGHT - PROP_SIZE) / SNAKE_SIZE);
        position.x = disX(gen) * SNAKE_SIZE;
        position.y = disY(gen) * SNAKE_SIZE;
    }
};

class Enemy {
public:
    Point position;
    int direction;
    int speed;
    bool isDead;
    Enemy(int startX, int startY) : position(startX, startY), direction(1), speed(20), isDead(false) {}

    void chase(Snake& snake) {
        // 简单追逐逻辑，未使用A*算法
        if (snake.body[0].x > position.x) {
            direction = 1;
        } else if (snake.body[0].x < position.x) {
            direction = 3;
        } else if (snake.body[0].y > position.y) {
            direction = 2;
        } else if (snake.body[0].y < position.y) {
            direction = 0;
        }
        switch (direction) {
            case 0: position.y -= speed; break;
            case 1: position.x += speed; break;
            case 2: position.y += speed; break;
            case 3: position.x -= speed; break;
        }
    }
};

class Bullet {
public:
    Point position;
    int direction;
    bool isActive;
    Bullet() : isActive(false) {}

    void fire(Snake& snake) {
        position = snake.body[0];
        direction = snake.direction;
        isActive = true;
    }

    void move() {
        switch (direction) {
            case 0: position.y -= 20; break;
            case 1: position.x += 20; break;
            case 2: position.y += 20; break;
            case 3: position.x -= 20; break;
        }
        if (position.x < 0 || position.x >= WIDTH || position.y < 0 || position.y >= HEIGHT) {
            isActive = false;
        }
    }
};

bool isCollision(Point a, Point b, int sizeA, int sizeB) {
    return (a.x < b.x + sizeB && a.x + sizeA > b.x &&
            a.y < b.y + sizeB && a.y + sizeA > b.y);
}

int main() {
    cv::Mat frame(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    Snake snake;
    Food food;
    Prop prop;
    std::vector<Enemy> enemies;
    Bullet bullet;
    std::clock_t lastEnemySpawnTime = std::clock();
    std::clock_t propActivationTime = 0;

    cv::namedWindow("Snake Game", cv::WINDOW_NORMAL);
    cv::resizeWindow("Snake Game", WIDTH, HEIGHT);

    while (true) {
        frame.setTo(cv::Scalar(0, 0, 0));

        for (const auto& part : snake.body) {
            cv::rectangle(frame, cv::Rect(part.x, part.y, SNAKE_SIZE, SNAKE_SIZE), cv::Scalar(0, 255, 0), -1);
        }

        cv::rectangle(frame, cv::Rect(food.position.x, food.position.y, FOOD_SIZE, FOOD_SIZE),
                      [&food]() {
                          switch (food.type) {
                              case 0: return cv::Scalar(255, 255, 255);
                              case 1: return cv::Scalar(0, 255, 0);
                              case 2: return cv::Scalar(255, 0, 0);
                              case 3: return cv::Scalar(0, 0, 255);
                              default: return cv::Scalar(255, 255, 255);
                          }
                      }(), -1);

        if (prop.active) {
            cv::rectangle(frame, cv::Rect(prop.position.x, prop.position.y, PROP_SIZE, PROP_SIZE),
                          [&prop]() {
                              switch (prop.type) {
                                  case 0: return cv::Scalar(255, 215, 0);
                                  case 1: return cv::Scalar(0, 255, 255);
                                  case 2: return cv::Scalar(128, 0, 128);
                                  default: return cv::Scalar(255, 255, 255);
                              }
                          }(), -1);
        }

        for (const auto& enemy : enemies) {
            if (!enemy.isDead) {
                cv::rectangle(frame, cv::Rect(enemy.position.x, enemy.position.y, ENEMY_SIZE, ENEMY_SIZE), cv::Scalar(255, 0, 0), -1);
            }
        }

        if (bullet.isActive) {
            cv::rectangle(frame, cv::Rect(bullet.position.x, bullet.position.y, BULLET_SIZE, BULLET_SIZE), cv::Scalar(255, 255, 0), -1);
        }

        cv::putText(frame, "Score: " + std::to_string(snake.score), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, "Lives: " + std::to_string(snake.lives), cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

        cv::imshow("Snake Game", frame);

        int key = cv::waitKey(1000 / snake.speed);
        if (key == 27) break;
        if (key == cv::KeyCode::UP && snake.direction!= 2) snake.direction = 0;
        if (key == cv::KeyCode::DOWN && snake.direction!= 0) snake.direction = 2;
        if (key == cv::KeyCode::LEFT && snake.direction!= 1) snake.direction = 3;
        if (key == cv::KeyCode::RIGHT && snake.direction!= 3) snake.direction = 1;
        if (key == 'q' && snake.score >= 30) {
            bullet.fire(snake);
            snake.score -= 30;
        }

        snake.move();
        if (snake.checkSelfCollision() || snake.body[0].x < 0 || snake.body[0].x >= WIDTH || snake.body[0].y < 0 || snake.body[0].y >= HEIGHT) {
            snake.lives--;
            if (snake.lives == 0) break;
            snake.body.clear();
            snake.body.emplace_back(WIDTH / 2, HEIGHT / 2);
            snake.direction = 1;
            snake.speed = 10;
        }

        if (isCollision(snake.body[0], food.position, SNAKE_SIZE, FOOD_SIZE)) {
            snake.score += [&food]() {
                switch (food.type) {
                    case 0: return 10;
                    case 1: snake.speed += 5; return 20;
                    case 2: snake.speed -= 5; return 5;
                    case 3: snake.lives++; return 15;
                    default: return 0;
                }
            }();
            food.generatePosition();
            food.type = std::rand() % 4;
            snake.body.push_back(snake.body.back());
        }

        if (prop.active && (std::clock() - propActivationTime) / (double)CLOCKS_PER_SEC >= 15) {
            prop.active = false;
        }

        if (!prop.active && isCollision(snake.body[0], prop.position, SNAKE_SIZE, PROP_SIZE)) {
            prop.active = true;
            propActivationTime = std::clock();
            prop.generatePosition();
            prop.type = std::rand() % 3;
        }

        if (bullet.isActive) {
            bullet.move();
            for (auto& enemy : enemies) {
                if (!enemy.isDead && isCollision(bullet.position, enemy.position, BULLET_SIZE, ENEMY_SIZE)) {
                    enemy.isDead = true;
                    bullet.isActive = false;
                    snake.score += 50;
                }
            }
        }

        if ((std::clock() - lastEnemySpawnTime) / (double)CLOCKS_PER_SEC >= 20) {
            enemies.emplace_back(0, 0);
            lastEnemySpawnTime = std::clock();
        }

        for (auto& enemy : enemies) {
            if (!enemy.isDead) {
                enemy.chase(snake);
                if (isCollision(snake.body[0], enemy.position, SNAKE_SIZE, ENEMY_SIZE)) {
                    int enemyCount = static_cast<int>(enemies.size());
                    for (auto& e : enemies) {
                        e.isDead = true;
                    }
                    snake.lives -= enemyCount;
                    if (snake.lives <= 0) break;
                }
            }
        }

        if (snake.lives <= 0) break;

        enemies.erase(std::remove_if(enemies.begin(), enemies.end(), [](const Enemy& e) { return e.isDead; }), enemies.end());
    }

    cv::destroyAllWindows();
    return 0;
}
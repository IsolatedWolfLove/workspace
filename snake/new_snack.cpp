#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <random>
#include <chrono>
#include <thread>

using namespace cv;
using namespace std;

const int WIDTH = 800;
const int HEIGHT = 600;
const int CELL_SIZE = 20;
const int INITIAL_LIVES = 3;
const int ENEMY_SPAWN_INTERVAL = 20; // in seconds
const int POWERUP_DURATION = 15; // in seconds

enum FoodType { NORMAL, SPEED_UP, SPEED_DOWN, LIFE };
enum PowerupType { INVINCIBLE, WALL_PASS, DOUBLE_SCORE };

struct Point {
    int x, y;
    Point(int x = 0, int y = 0) : x(x), y(y) {}
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

struct Food {
    cv::Point position;
    FoodType type;
    Scalar color;
};

struct Powerup {
    cv::Point position;
    PowerupType type;
    Scalar color;
};

struct Enemy {
    cv::Point position;
    cv::Point direction;
    bool isAlive;
};

class Snake {
public:
    deque<cv::Point> body;
    cv::Point direction;
    bool invincible;
    bool canPassWalls;
    int scoreMultiplier;

    Snake() : direction(1, 0), invincible(false), canPassWalls(false), scoreMultiplier(1) {
        body.push_back(cv::Point(WIDTH / 2, HEIGHT / 2));
    }

    void move() {
        cv::Point newHead = body.front() + direction;
        body.push_front(newHead);
        body.pop_back();
    }

    void grow() {
        body.push_back(body.back());
    }

    bool collidesWith(const cv::Point& point) const {
        for (const auto& segment : body) {
            if (segment == point) return true;
        }
        return false;
    }

    bool collidesWithSelf() const {
        for (size_t i = 1; i < body.size(); ++i) {
            if (body[i] == body.front()) return true;
        }
        return false;
    }

    bool isOutOfBounds() const {
        cv::Point head = body.front();
        return head.x < 0 || head.x >= WIDTH || head.y < 0 || head.y >= HEIGHT;
    }
};

class Game {
private:
    Snake snake;
    Food food;
    Powerup powerup;
    vector<Enemy> enemies;
    int score;
    int lives;
    bool gameOver;
    chrono::time_point<chrono::system_clock> lastEnemySpawnTime;

    void spawnFood() {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> disX(0, WIDTH / CELL_SIZE - 1);
        uniform_int_distribution<> disY(0, HEIGHT / CELL_SIZE - 1);

        food.position = cv::Point(disX(gen) * CELL_SIZE, disY(gen) * CELL_SIZE);
        food.type = static_cast<FoodType>(disX(gen) % 4);
        switch (food.type) {
            case NORMAL: food.color = Scalar(255, 255, 255); break;
            case SPEED_UP: food.color = Scalar(0, 255, 0); break;
            case SPEED_DOWN: food.color = Scalar(0, 0, 255); break;
            case LIFE: food.color = Scalar(255, 0, 0); break;
        }
    }

    void spawnPowerup() {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> disX(0, WIDTH / CELL_SIZE - 1);
        uniform_int_distribution<> disY(0, HEIGHT / CELL_SIZE - 1);

        powerup.position = cv::Point(disX(gen) * CELL_SIZE, disY(gen) * CELL_SIZE);
        powerup.type = static_cast<PowerupType>(disX(gen) % 3);
        switch (powerup.type) {
            case INVINCIBLE: powerup.color = Scalar(0, 255, 255); break;
            case WALL_PASS: powerup.color = Scalar(255, 255, 0); break;
            case DOUBLE_SCORE: powerup.color = Scalar(255, 0, 255); break;
        }
    }

    void spawnEnemy() {
        Enemy enemy;
        enemy.position = cv::Point(0, 0);
        enemy.direction = cv::Point(1, 0);
        enemy.isAlive = true;
        enemies.push_back(enemy);
        lastEnemySpawnTime = chrono::system_clock::now();
    }

    void updateEnemies() {
        for (auto& enemy : enemies) {
            if (enemy.isAlive) {
                enemy.position = enemy.position + enemy.direction;
                if (enemy.position.x < 0 || enemy.position.x >= WIDTH || enemy.position.y < 0 || enemy.position.y >= HEIGHT) {
                    enemy.isAlive = false;
                }
            }
        }
    }

    void checkCollisions() {
        if (snake.collidesWith(food.position)) {
            switch (food.type) {
                case NORMAL: score += 10; break;
                case SPEED_UP: score += 20; break;
                case SPEED_DOWN: score += 5; break;
                case LIFE: lives++; break;
            }
            spawnFood();
            snake.grow();
        }

        if (snake.collidesWith(powerup.position)) {
            switch (powerup.type) {
                case INVINCIBLE: snake.invincible = true; break;
                case WALL_PASS: snake.canPassWalls = true; break;
                case DOUBLE_SCORE: snake.scoreMultiplier = 2; break;
            }
            spawnPowerup();
        }

        for (auto& enemy : enemies) {
            if (enemy.isAlive && snake.collidesWith(enemy.position)) {
                if (!snake.invincible) {
                    lives -= enemies.size();
                    enemies.clear();
                    if (lives <= 0) gameOver = true;
                }
            }
        }
    }

public:
    Game() : score(0), lives(INITIAL_LIVES), gameOver(false) {
        spawnFood();
        spawnPowerup();
        lastEnemySpawnTime = chrono::system_clock::now();
    }

    void run() {
        Mat frame(HEIGHT, WIDTH, CV_8UC3, Scalar(0, 0, 0));
        namedWindow("Snake Game", WINDOW_AUTOSIZE);

        while (!gameOver) {
            frame.setTo(Scalar(0, 0, 0));

            // Draw food
            rectangle(frame, Rect(food.position.x, food.position.y, CELL_SIZE, CELL_SIZE), food.color, FILLED);

            // Draw powerup
            drawMarker(frame, cv::Point(powerup.position.x + CELL_SIZE / 2, powerup.position.y + CELL_SIZE / 2), powerup.color, MARKER_STAR, CELL_SIZE, 2);

            // Draw snake
            for (const auto& segment : snake.body) {
                rectangle(frame, Rect(segment.x, segment.y, CELL_SIZE, CELL_SIZE), Scalar(0, 255, 0), FILLED);
            }

            // Draw enemies
            for (const auto& enemy : enemies) {
                if (enemy.isAlive) {
                    drawMarker(frame, cv::Point(enemy.position.x + CELL_SIZE / 2, enemy.position.y + CELL_SIZE / 2), Scalar(0, 0, 255), MARKER_TRIANGLE_UP, CELL_SIZE, 2);
                }
            }

            imshow("Snake Game", frame);

            int key = waitKey(100);
            switch (key) {
                case 'w': snake.direction = cv::Point(0, -1); break;
                case 's': snake.direction = cv::Point(0, 1); break;
                case 'a': snake.direction = cv::Point(-1, 0); break;
                case 'd': snake.direction = cv::Point(1, 0); break;
                case 'q': if (score >= 30) { score -= 30; enemies.erase(enemies.begin()); } break;
                case 27: gameOver = true; break; // ESC key
            }

            snake.move();
            checkCollisions();
            updateEnemies();

            if (chrono::duration_cast<chrono::seconds>(chrono::system_clock::now() - lastEnemySpawnTime).count() >= ENEMY_SPAWN_INTERVAL) {
                spawnEnemy();
            }

            if (snake.isOutOfBounds() && !snake.canPassWalls) {
                gameOver = true;
            }

            if (snake.collidesWithSelf()) {
                gameOver = true;
            }
        }

        destroyWindow("Snake Game");
        cout << "Game Over! Final Score: " << score << endl;
    }
};

int main() {
    Game game;
    game.run();
    return 0;
}
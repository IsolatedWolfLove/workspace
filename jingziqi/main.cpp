#include <iostream>
#include <opencv2/opencv.hpp>

class JingZiqi {
public:
    JingZiqi(int size_ = 1023) {
        size = size_ / 3;
        image = cv::Mat(size_, size_, CV_8UC3, cv::Scalar(0, 0, 0));
        draw_Chessboard();
        cv::namedWindow("image");
        cv::setMouseCallback("image", on_mouse, this);
        cv::imshow("image", image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    void draw_Chessboard() {
        for (int i = 1; i < 3; i++) {
            cv::line(image, cv::Point(size * i, 0), cv::Point(size * i, size), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
            cv::line(image, cv::Point(0, size * i), cv::Point(size, size * i), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        }
    }

    static void on_mouse(int event, int x, int y, int flags, void* userdata) {
        if (event == cv::EVENT_LBUTTONDOWN) {
            JingZiqi* jz = reinterpret_cast<JingZiqi*>(userdata);
            std::pair<int, int> mouse_pos = jz->mouse_detect(x, y);
            if (mouse_pos.first != -1 && mouse_pos.second != -1 && !jz->is_occupied(mouse_pos.first, mouse_pos.second)) {
                if (jz->is_circle) {
                    jz->draw_circle(jz->image, mouse_pos.first, mouse_pos.second);
                } else {
                    jz->draw_x(jz->image, mouse_pos.first, mouse_pos.second);
                }
                jz->is_circle = !jz->is_circle; // Toggle between circle and X
            }
        }
    }

    std::pair<int, int> mouse_detect(int x, int y) {
        int point_start = size / 3;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if ((x > point_start * i) && (x < point_start * (i + 1)) &&
                    (y > point_start * j) && (y < point_start * (j + 1))) {
                    std::cout << "mouse_x: " << i << " mouse_y: " << j << std::endl;
                    return std::make_pair(i, j);
                }
            }
        }
        std::cout << "error";
        return std::make_pair(-1, -1);
    }

    bool is_occupied(int x, int y) {
        // 这里可以根据实际情况检查当前位置是否有棋子，简单起见，我们可以假设每次都标记为未占用
        return false;
    }

    void draw_circle(cv::Mat& image, int x, int y) {
        cv::circle(image, cv::Point(size * x / 2, size * y / 2), 5, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    }

    void draw_x(cv::Mat& image, int x1, int y1) {
        cv::line(image, cv::Point(size * x1, size * y1), cv::Point(size * (x1 + 1), size * (y1 + 1)), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        cv::line(image, cv::Point(size * x1, size * (y1 + 1)), cv::Point(size * (x1 + 1), size * y1), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    }

private:
    int size;
    bool is_circle = true; // Start with drawing a circle
    cv::Mat image;
};

int main() {
    JingZiqi jingziqi(1023);
    return 0;
}
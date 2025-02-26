#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <numeric>
const int BOARD_ROWS = 3;
const int BOARD_COLS = 3;
const int BOARD_SIZE = BOARD_ROWS * BOARD_COLS;

class TicTacToeAI {
public:
    TicTacToeAI() {
        state = std::vector<std::vector<int>>(BOARD_ROWS, std::vector<int>(BOARD_COLS, 0));
        winner = 0;
        end = false;
        load_policy();
    }

    void load_policy() {
        std::ifstream file("policy_second.txt");
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            long long hash_val;
            float estimation;
            iss >> hash_val >> estimation;
            estimations[hash_val] = estimation;
        }
    }

    long long hash_state() {
        long long hash_val = 0;
        for (int i = 0; i < BOARD_ROWS; ++i) {
            for (int j = 0; j < BOARD_COLS; ++j) {
                hash_val = hash_val * 3 + state[i][j] + 1;
            }
        }
        return hash_val;
    }

    bool is_end() {
        std::vector<int> results;
        for (int i = 0; i < BOARD_ROWS; ++i) {
            results.push_back(std::accumulate(state[i].begin(), state[i].end(), 0));
        }
        for (int j = 0; j < BOARD_COLS; ++j) {
            int sum = 0;
            for (int i = 0; i < BOARD_ROWS; ++i) {
                sum += state[i][j];
            }
            results.push_back(sum);
        }
        int trace = 0;
        for (int i = 0; i < BOARD_ROWS; ++i) {
            trace += state[i][i];
        }
        results.push_back(trace);
        int reverse_trace = 0;
        for (int i = 0; i < BOARD_ROWS; ++i) {
            reverse_trace += state[i][BOARD_COLS - 1 - i];
        }
        results.push_back(reverse_trace);

        for (int result : results) {
            if (result == 3) {
                winner = 1;
                end = true;
                return true;
            }
            if (result == -3) {
                winner = -1;
                end = true;
                return true;
            }
        }
        if (std::count_if(state.begin(), state.end(), [](const std::vector<int>& row) {
            return std::count(row.begin(), row.end(), 0) == 0;
        }) == BOARD_ROWS) {
            winner = 0;
            end = true;
            return true;
        }
        end = false;
        return false;
    }

    void print_state() {
        for (int i = 0; i < BOARD_ROWS; ++i) {
            std::cout << "-------------" << std::endl;
            std::string out = "| ";
            for (int j = 0; j < BOARD_COLS; ++j) {
                std::string token;
                if (state[i][j] == 1) {
                    token = "*";
                } else if (state[i][j] == -1) {
                    token = "x";
                } else {
                    token = "0";
                }
                out += token + " | ";
            }
            std::cout << out << std::endl;
        }
        std::cout << "-------------" << std::endl;
    }

    std::pair<int, int> ai_move() {
        std::vector<std::pair<long long, std::pair<int, int>>> values;
        for (int i = 0; i < BOARD_ROWS; ++i) {
            for (int j = 0; j < BOARD_COLS; ++j) {
                if (state[i][j] == 0) {
                    std::vector<std::vector<int>> new_state = state;
                    new_state[i][j] = -1;
                    long long hash_val = hash_state(new_state);
                    values.push_back({estimations.count(hash_val) ? estimations[hash_val] : 0.5, {i, j}});
                }
            }
        }
        std::sort(values.begin(), values.end(), [](const std::pair<long long, std::pair<int, int>>& a, const std::pair<long long, std::pair<int, int>>& b) {
            return a.first > b.first;
        });
        return values[0].second;
    }

    std::pair<int, int> play(char human_input) {
        std::vector<char> keys = {'q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c'};
        int data = std::find(keys.begin(), keys.end(), human_input) - keys.begin();
        int i = data / BOARD_COLS;
        int j = data % BOARD_COLS;

        if (state[i][j] != 0) {
            std::cout << "Invalid move! Position already occupied." << std::endl;
            return {-1, -1};
        }

        state[i][j] = 1;

        print_state();

        if (is_end()) {
            if (winner == 1) {
                std::cout << "You win!" << std::endl;
            } else if (winner == -1) {
                std::cout << "You lose!" << std::endl;
            } else {
                std::cout << "It's a tie!" << std::endl;
            }
            return {-1, -1};
        }

        std::pair<int, int> ai_move_pos = ai_move();
        state[ai_move_pos.first][ai_move_pos.second] = -1;

        print_state();

        if (is_end()) {
            if (winner == 1) {
                std::cout << "You win!" << std::endl;
            } else if (winner == -1) {
                std::cout << "You lose!" << std::endl;
            } else {
                std::cout << "It's a tie!" << std::endl;
            }
            return {-1, -1};
        }
        return ai_move_pos;
    }

private:
    std::vector<std::vector<int>> state;
    int winner;
    bool end;
    std::unordered_map<long long, float> estimations;

    long long hash_state(const std::vector<std::vector<int>>& state) {
        long long hash_val = 0;
        for (int i = 0; i < BOARD_ROWS; ++i) {
            for (int j = 0; j < BOARD_COLS; ++j) {
                hash_val = hash_val * 3 + state[i][j] + 1;
            }
        }
        return hash_val;
    }
};

int main() {
    TicTacToeAI ai;
    while (true) {
        char human_input;
        std::cout << "Input your position (q, w, e, a, s, d, z, x, c): ";
        std::cin >> human_input;
        std::pair<int, int> ai_move = ai.play(human_input);
        if (ai_move.first == -1) {
            break;  // Game is over
        }
    }
    return 0;
}
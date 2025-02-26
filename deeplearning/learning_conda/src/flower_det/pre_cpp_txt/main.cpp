#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
namespace fs = std::filesystem;
int main() {

  fs::path dir_path_root = "/home/ccy/workspace/deeplearning/learning_conda/"
                           "src/flower_det/data/flowers";
  //   std::vector<std::string> labels = {"daisy", "dandelion", "roses",
  //   "sunflowers", "tulips"};
  std::string fileNames = "/home/ccy/workspace/deeplearning/learning_conda/src/"
                          "flower_det/data/label.txt";
  std::ofstream ofm04(fileNames, std::ofstream::out);
  // 创建一个目录迭代器
  for (const auto &entry : fs::directory_iterator(dir_path_root)) {
    // 打印文件或目录的路径
    std::string fileName = entry.path().filename().string();
    std::string dir_path = dir_path_root.string() + "/" + fileName;
    for (const auto &entry : fs::directory_iterator(dir_path)) {
      if (entry.path().extension() == ".jpg") {
        ofm04 << fileName << "/"<<entry.path().filename().string() << " " << fileName
              << std::endl;
      }
    }
  }
}

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
namespace fs = std::filesystem;
int main() {

  fs::path dir_path_root =
      "/home/ccy/workspace/deeplearning/learning_conda/src/fruit_det/fruit_141/"
      "fruits-360_dataset_100x100/fruits-360/Training";
  //   std::vector<std::string> labels = {"daisy", "dandelion", "roses",
  //   "sunflowers", "tulips"};
  std::string fileNames =
      "/home/ccy/workspace/deeplearning/learning_conda/src/fruit_det/fruit_141/"
      "fruits-360_dataset_100x100/fruits-360/label.txt";
  std::ofstream ofm04(fileNames, std::ofstream::out);
  int i=0;
  // 创建一个目录迭代器
  for (const auto &entry : fs::directory_iterator(dir_path_root)) {

    // 打印文件或目录的路径
    std::string fileName = entry.path().filename().string();
    std::string dir_path = dir_path_root.string() + "/" + fileName;
    // for (const auto &entry : fs::directory_iterator(dir_path)) {
    //   if (entry.path().extension() == ".jpg") {
    //     ofm04 << fileName << "/" << entry.path().filename().string() << ","
    //           << fileName << std::endl;
    //   }
    // }
    //std::cout<<"                    if label==\""<<fileName<<"\":"<<"\n                        label="<<i<<std::endl;
    std::cout <<  i << ": \"" <<  fileName << "\","<<std::endl;
    i++;
  }
}

#ifndef DATASET_HPP
#define DATASET_HPP

#include "tensor.hpp"
#include <string>
#include <vector>
#include <map>

class Dataset {
public:
    std::vector<std::string> image_paths;
    std::vector<int> labels;
    std::map<std::string, int> class_map;

    void load_structure(const std::string& root_path);


    void get_batch(const std::vector<int>& indices, Tensor& X, Tensor& Y);
};

#endif

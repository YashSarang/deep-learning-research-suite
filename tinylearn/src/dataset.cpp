#include "../include/dataset.hpp"
#include <filesystem>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

void Dataset::load_structure(const std::string& root_path) {
    int current_label = 0;
    if (!fs::exists(root_path)) {
        std::cerr << "Error: Directory " << root_path << " does not exist." << std::endl;
        exit(1);
    }

    for (const auto& entry : fs::directory_iterator(root_path)) {
        if (entry.is_directory()) {
            std::string class_name = entry.path().filename().string();
            class_map[class_name] = current_label;

            for (const auto& img_entry : fs::directory_iterator(entry.path())) {
                std::string ext = img_entry.path().extension().string();
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                    image_paths.push_back(img_entry.path().string());
                    labels.push_back(current_label);
                }
            }
            current_label++;
        }
    }
}

void Dataset::get_batch(const std::vector<int>& indices, Tensor& X, Tensor& Y) {
    int batch_size = indices.size();
    X = Tensor(batch_size, 3, 32, 32); 
    Y = Tensor(batch_size, 1, 1, 1);

    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        int idx = indices[i];

        cv::Mat img = cv::imread(image_paths[idx]);
        if (img.empty()) continue; 

        if (img.rows != 32 || img.cols != 32) {
            cv::resize(img, img, cv::Size(32, 32));
        }

        for (int r = 0; r < 32; ++r) {
            for (int c = 0; c < 32; ++c) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(r, c);
                X(i, 0, r, c) = pixel[2] / 255.0f; 
                X(i, 1, r, c) = pixel[1] / 255.0f; 
                X(i, 2, r, c) = pixel[0] / 255.0f; 
            }
        }
        Y.data[i] = (float)labels[idx];
    }
}

#include "include/tensor.hpp"
#include "include/dataset.hpp"
#include "include/layers.hpp"
#include <iostream>
#include <fstream>  
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip> 


struct DualStream {
    std::ofstream file;

    DualStream(const std::string& filename) {
        file.open(filename, std::ios::out);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open " << filename << " for writing." << std::endl;
        }
    }

    ~DualStream() {
        if (file.is_open()) file.close();
    }

    template <typename T>
    DualStream& operator<<(const T& data) {
        std::cout << data;
        if (file.is_open()) file << data;
        return *this;
    }

    DualStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        std::cout << manip;
        if (file.is_open()) file << manip;
        return *this;
    }
};

std::string format_metric(long long n) {
    if (n >= 1e6) {
        char buf[16];
        sprintf(buf, "%.2fM", n / 1e6);
        return std::string(buf);
    } else if (n >= 1e3) {
        char buf[16];
        sprintf(buf, "%.2fK", n / 1e3);
        return std::string(buf);
    }
    return std::to_string(n);
}

int main(int argc, char** argv) {
    // 1. Initialize Logger
    DualStream logger("logs.txt");

    if (argc < 2) {
        logger << "Usage: ./cnn_scratch <path_to_dataset_folder>" << std::endl;
        return 1;
    }
    std::string data_path = argv[1];

    
    int epochs = 30;
    int batch_size = 64;
    float learning_rate = 0.01f;
    std::string optimizer_name = "SGD";
    std::string init_type = "He Normal (Scale: sqrt(2/fan_in))";

    
    logger << "Loading Dataset from " << data_path << "..." << std::endl;
    Dataset ds;

    auto t_load_start = std::chrono::high_resolution_clock::now();
    ds.load_structure(data_path);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double>(t_load_end - t_load_start).count();

    if (ds.labels.empty()) {
        logger << "Error: No images found. Check path." << std::endl;
        return 1;
    }

    std::vector<int> indices(ds.labels.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 g(1234);
    std::shuffle(indices.begin(), indices.end(), g);

    size_t split = (size_t)(indices.size() * 0.8);
    std::vector<int> train_idx(indices.begin(), indices.begin() + split);
    std::vector<int> val_idx(indices.begin() + split, indices.end());

    
    std::vector<Layer*> model;

    model.push_back(new Conv2D(3, 16, 3, 1, 1));
    model.push_back(new ReLU());
    model.push_back(new MaxPool2D(2, 2));

    model.push_back(new Conv2D(16, 32, 3, 1, 1));
    model.push_back(new ReLU());
    model.push_back(new MaxPool2D(2, 2));


    model.push_back(new Linear(32 * 8 * 8, 100)); 

    CrossEntropyLoss criterion;


    long long total_params = 0;
    long long total_macs = 0;

    logger << "\n================================================================================" << std::endl;
    logger << "                                MODEL SUMMARY                                   " << std::endl;
    logger << "================================================================================" << std::endl;
    logger << std::left << std::setw(15) << "Layer"
    << std::setw(15) << "Params"
    << std::setw(15) << "MACs" << std::endl;
    logger << "--------------------------------------------------------------------------------" << std::endl;

    Tensor t(1, 3, 32, 32);
    for(auto l : model) {
        t = l->forward(t);
        if (dynamic_cast<Linear*>(l) && t.c > 1) t.flatten();

        long long p = l->get_params();
        long long m = l->get_macs();
        total_params += p;
        total_macs += m;

        logger << std::left << std::setw(15) << l->get_name()
        << std::setw(15) << p
        << std::setw(15) << m << std::endl;
    }

    long long total_flops = total_macs * 2;

    logger << "================================================================================" << std::endl;
    logger << "Dataset Loading Time : " << std::fixed << std::setprecision(4) << load_time << " s" << std::endl;
    logger << "Total Images         : " << indices.size() << " (Train: " << train_idx.size() << ", Val: " << val_idx.size() << ")" << std::endl;
    logger << "Total Parameters     : " << total_params << " (" << format_metric(total_params) << ")" << std::endl;
    logger << "MACs per Input       : " << total_macs << " (" << format_metric(total_macs) << ")" << std::endl;
    logger << "FLOPs per Input      : " << total_flops << " (" << format_metric(total_flops) << ")" << std::endl;
    logger << "Optimizer            : " << optimizer_name << " (LR=" << learning_rate << ")" << std::endl;
    logger << "Initialization       : " << init_type << std::endl;
    logger << "================================================================================\n" << std::endl;


    logger << std::left
    << std::setw(6)  << "Epoch"
    << std::setw(12) << "Loss"
    << std::setw(12) << "Acc(%)"
    << std::setw(12) << "Time(s)"
    << std::setw(12) << "Params"
    << std::setw(12) << "Tot MACs"
    << std::setw(12) << "Tot FLOPs"
    << std::endl;
    logger << std::string(85, '-') << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto t_epoch_start = std::chrono::high_resolution_clock::now();

        std::shuffle(train_idx.begin(), train_idx.end(), g);
        float epoch_loss = 0;
        int num_batches = 0;

        for (size_t i = 0; i < train_idx.size(); i += batch_size) {
            std::vector<int> batch_idx;
            for(size_t j = i; j < std::min(i + batch_size, train_idx.size()); ++j)
                batch_idx.push_back(train_idx[j]);

            Tensor X, Y;
            ds.get_batch(batch_idx, X, Y);

            Tensor act = X;
            for (auto l : model) {
                act = l->forward(act);
                if (dynamic_cast<Linear*>(l) && act.c > 1) act.flatten();
            }

            float loss = criterion.forward(act, Y);
            epoch_loss += loss;
            num_batches++;

            Tensor grad = criterion.backward(Y);
            for (auto it = model.rbegin(); it != model.rend(); ++it) {
                grad = (*it)->backward(grad);
            }

            for (auto l : model) l->update(learning_rate);

            if (i % (batch_size * 10) == 0) { std::cout << "." << std::flush; }
        }
        std::cout << "\r"; 

        int correct = 0;
        size_t val_limit = val_idx.size();
        for (size_t i = 0; i < val_limit; i += batch_size) {
            std::vector<int> batch_idx;
            for(size_t j = i; j < std::min(i + batch_size, val_limit); ++j)
                batch_idx.push_back(val_idx[j]);

            Tensor X, Y;
            ds.get_batch(batch_idx, X, Y);

            Tensor act = X;
            for (auto l : model) {
                act = l->forward(act);
                if (dynamic_cast<Linear*>(l) && act.c > 1) act.flatten();
            }

            // Argmax
            for(int b=0; b<act.b; ++b) {
                float max_v = -1e9;
                int max_i = -1;
                int num_classes = act.w;
                for(int k=0; k<num_classes; ++k) {
                    if(act.data[b*num_classes+k] > max_v) {
                        max_v = act.data[b*num_classes+k];
                        max_i = k;
                    }
                }
                if(max_i == (int)Y.data[b]) correct++;
            }
        }

        float accuracy = (float)correct / val_limit * 100.0f;
        float avg_loss = epoch_loss / num_batches;

        auto t_epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_duration = std::chrono::duration<double>(t_epoch_end - t_epoch_start).count();

        logger << std::left
        << std::setw(6)  << (epoch + 1)
        << std::setw(12) << std::fixed << std::setprecision(4) << avg_loss
        << std::setw(12) << std::setprecision(2) << accuracy
        << std::setw(12) << std::setprecision(2) << epoch_duration
        << std::setw(12) << format_metric(total_params)
        << std::setw(12) << format_metric(total_macs)
        << std::setw(12) << format_metric(total_flops)
        << std::endl;

        if ((epoch + 1) % 5 == 0) {
            learning_rate *= 0.5f;
        }
    }

    for(auto l : model) delete l;
    return 0;
}

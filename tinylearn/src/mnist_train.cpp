#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>
#include "tensor.h"
#include "model.h"
#include "data_loader.h"
#include "utils.h"

class SGD {
public:
    float lr;
    float momentum;
    float weight_decay;
    std::vector<std::vector<float>> velocity;

    SGD(float lr_=1e-2f, float momentum_=0.9f, float weight_decay_=5e-4f)
        : lr(lr_), momentum(momentum_), weight_decay(weight_decay_) {}

    void init(const std::vector<std::pair<std::vector<float>*, std::vector<float>*>> &params) {
        velocity.clear();
        velocity.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            velocity[i].assign(params[i].first->size(), 0.0f);
        }
    }

    void step(const std::vector<std::pair<std::vector<float>*, std::vector<float>*>> &params) {
        for (size_t i = 0; i < params.size(); ++i) {
            auto &p = *params[i].first;
            auto &g = *params[i].second;
            auto &v = velocity[i];
            for (size_t j = 0; j < p.size(); ++j) {
                float grad = g[j] + weight_decay * p[j];
                v[j] = momentum * v[j] + (1.0f - momentum) * grad;
                p[j] -= lr * v[j];
                g[j] = 0.0f;
            }
        }
    }
};

long calculate_total_macs(SimpleCNN& model, int input_h, int input_w) {

    long m = 0;

    m += macs(model.conv1.out_channels, model.conv1.in_channels, model.conv1.kernel, 
              model.conv1.stride, model.conv1.pad, input_h, input_w);

    int h1 = (input_h - model.conv1.kernel + 2 * model.conv1.pad) / model.conv1.stride + 1;
    int w1 = (input_w - model.conv1.kernel + 2 * model.conv1.pad) / model.conv1.stride + 1;

    m += macs(model.conv2.out_channels, model.conv2.in_channels, model.conv2.kernel,
              model.conv2.stride, model.conv2.pad, h1, w1);

    int h2 = (h1 - model.conv2.kernel + 2 * model.conv2.pad) / model.conv2.stride + 1;
    int w2 = (w1 - model.conv2.kernel + 2 * model.conv2.pad) / model.conv2.stride + 1;

    int h_pool = h2 / 2;
    int w_pool = w2 / 2;

    int flat_size = model.conv2.out_channels * h_pool * w_pool;

    if (model.fc1) {
        m += macs_linear(model.fc1->in_features, model.fc1->out_features, 1, 0);
    } else {

        m += macs_linear(flat_size, 64, 1, 0);
    }

    m += macs_linear(model.fc2.in_features, model.fc2.out_features, 1, 0);

    return m;
}

int main()
{

    std::ofstream log_file("training_log.txt");
    if (!log_file.is_open()) {
        std::cerr << "Failed to create log file." << std::endl;
        return -1;
    }
    log_file << "--- Model Training Log ---" << std::endl;

    std::string path = "data";
    ImageFolderDataset train_dataset(path);

    if(train_dataset.samples.empty()) {
        std::cerr << "No samples found. Exiting." << std::endl;
        return -1;
    }

    SimpleCNN model(10);
    DataLoader loader(train_dataset, 128, true); 
    SGD optimizer(1e-2f, 0.9f);

    log_file << "\n[Initialization]" << std::endl;
    log_file << "Model: SimpleCNN" << std::endl;
    log_file << "Optimizer: SGD (lr=0.01, momentum=0.9, decay=5e-4)" << std::endl;
    log_file << "Dataset size: " << train_dataset.samples.size() << std::endl;

    log_file << "\n[Architecture]" << std::endl;
    log_file << "1. Conv2D (3 -> 8, k=3)" << std::endl;
    log_file << "2. ReLU" << std::endl;
    log_file << "3. Conv2D (8 -> 8, k=3)" << std::endl;
    log_file << "4. ReLU" << std::endl;
    log_file << "5. MaxPool2D (2x2)" << std::endl;
    log_file << "6. Linear (Flatten -> 64)" << std::endl;
    log_file << "7. ReLU" << std::endl;
    log_file << "8. Linear (64 -> 10)" << std::endl;

    auto param_pairs = model.get_param_grad_pairs();
    optimizer.init(param_pairs);

    double total_training_time = 0.0;
    int epochs = 2;

    log_file << "\n[Epoch Statistics]" << std::endl;
    log_file << std::left << std::setw(8) << "Epoch" 
             << std::setw(15) << "Loss" 
             << std::setw(15) << "Accuracy(%)" 
             << std::setw(15) << "Time(s)" 
             << std::setw(15) << "Params" 
             << std::setw(15) << "Total MACs" 
             << std::setw(15) << "Total FLOPs" << std::endl;

    float final_accuracy = 0.0f;
    float final_loss = 0.0f;
    int total_params = 0;
    long macs_per_sample = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto start = std::chrono::high_resolution_clock::now();
        loader.reset();
        float epoch_loss = 0.0f;
        int correct = 0;
        int total = 0;
        int batch_idx = 0;

        while (loader.has_next()) {
            auto batch = loader.next_batch();
            model.zero_grad();

            float batch_loss = 0.0f;
            for (Sample* sample : batch) {
                Tensor1D logits = model.forward(sample->image);

                Tensor1D grad_logits(logits.size);
                float loss = cross_entropy_loss_and_grad(logits, sample->label, grad_logits);
                batch_loss += loss;

                int pred = print_argmax(logits);
                if (pred == sample->label) correct++;
                total++;

                model.backward(grad_logits);
            }

            auto current_params = model.get_param_grad_pairs();
            if (current_params.size() != optimizer.velocity.size()) {
                optimizer.init(current_params);

                if (total_params == 0) {

                    total_params = model.conv1.num_params() + model.conv2.num_params() + 
                                   model.fc1->num_params() + model.fc2.num_params();

                    macs_per_sample = calculate_total_macs(model, 32, 32);
                }
            }

            optimizer.step(current_params);
            epoch_loss += batch_loss;
            batch_idx++;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        total_training_time += seconds;

        float avg_loss = epoch_loss / (float)train_dataset.samples.size();
        float acc = 100.0f * correct / total;
        final_accuracy = acc;
        final_loss = avg_loss;

        long total_epoch_macs = macs_per_sample * total;
        long total_epoch_flops = 2 * total_epoch_macs; 

        std::cout << "Epoch " << epoch << " finished. Loss: " << avg_loss 
                  << " Acc: " << acc << "% Time: " << seconds << "s\n";

        log_file << std::left << std::setw(8) << epoch 
                 << std::setw(15) << avg_loss 
                 << std::setw(15) << acc 
                 << std::setw(15) << seconds 
                 << std::setw(15) << total_params 
                 << std::setw(15) << total_epoch_macs 
                 << std::setw(15) << total_epoch_flops << std::endl;
    }

    log_file << "\n[Overall Statistics]" << std::endl;
    log_file << "Total Epochs: " << epochs << std::endl;
    log_file << "Final Accuracy: " << final_accuracy << "%" << std::endl;
    log_file << "Final Loss: " << final_loss << std::endl;
    log_file << "Total Training Time: " << total_training_time << " seconds" << std::endl;
    log_file << "Model Parameters: " << total_params << std::endl;
    log_file << "MACs per Inference: " << macs_per_sample << std::endl;
    log_file << "Training Throughput: " << (epochs * train_dataset.samples.size()) / total_training_time << " samples/sec" << std::endl;

    log_file.close();
    std::cout << "Training log saved to training_log.txt" << std::endl;
    model.save_parameters("cnn_weights.bin");

    return 0;
}
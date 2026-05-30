#pragma once

#include <iostream>
#include <filesystem>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <thread>
#include <atomic>
#include <mutex>
#include <opencv4/opencv2/opencv.hpp>   
#include "tensor.h" 

namespace fs = std::filesystem;

struct Sample
{
    Tensor image;
    int label;
    Sample(Tensor &&img, int lbl) : image(std::move(img)), label(lbl) {}
};

class ImageFolderDataset
{
public:
    std::vector<Sample> samples;

    ImageFolderDataset(const std::string &root_dir)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (!fs::exists(root_dir)) {
            std::cerr << "Directory " << root_dir << " not found." << std::endl;
            return;
        }

        struct PathLabel { std::string path; int label; };
        std::vector<PathLabel> all_paths;
        for (const auto &class_dir : fs::directory_iterator(root_dir))
        {
            if (!class_dir.is_directory()) continue;

            int label = std::stoi(class_dir.path().filename().string());

            for (const auto &img_path : fs::directory_iterator(class_dir))
            {
                all_paths.push_back({ img_path.path().string(), label });
            }
        }

        unsigned int hw = std::thread::hardware_concurrency();
        const unsigned int num_threads = hw == 0 ? 4 : hw;
        std::atomic<size_t> next_idx(0);
        std::mutex samples_mutex;
        samples.reserve(all_paths.size());

        auto worker = [&]() {
            while (true) {
                size_t i = next_idx.fetch_add(1);
                if (i >= all_paths.size()) break;
                const auto &pl = all_paths[i];

                cv::Mat img = cv::imread(pl.path, cv::IMREAD_COLOR);
                if (img.empty()) continue;

                cv::resize(img, img, cv::Size(32, 32));
                img.convertTo(img, CV_32F, 1.0 / 255.0);

                Tensor tensor(3, 32, 32);
                for (int h = 0; h < 32; ++h)
                {
                    for (int w = 0; w < 32; ++w)
                    {
                        cv::Vec3f pixel = img.at<cv::Vec3f>(h, w);
                        for (int c = 0; c < 3; ++c)
                            tensor(c, h, w) = pixel[c];
                    }
                }

                {
                    std::lock_guard<std::mutex> lg(samples_mutex);
                    samples.emplace_back(std::move(tensor), pl.label);
                }
            }
        };

        std::vector<std::thread> threads;
        for (unsigned int t = 0; t < num_threads; ++t)
            threads.emplace_back(worker);
        for (auto &th : threads) th.join();

        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        std::cout << "Dataset loaded in " << seconds << " seconds. Total samples: " << samples.size() << "\n";
    }
};

class DataLoader
{
public:
    ImageFolderDataset &dataset;
    size_t batch_size;
    bool shuffle;
    size_t index;
    std::vector<size_t> indices;

    DataLoader(ImageFolderDataset &ds, size_t bs, bool sh = true)
        : dataset(ds), batch_size(bs), shuffle(sh), index(0)
    {
        indices.resize(dataset.samples.size());
        std::iota(indices.begin(), indices.end(), 0);

        if (shuffle)
        {
            std::mt19937 rng(std::random_device{}());
            std::shuffle(indices.begin(), indices.end(), rng);
        }
    }

    bool has_next() const { return index < indices.size(); }

    std::vector<Sample *> next_batch()
    {
        size_t end = std::min(index + batch_size, indices.size());
        std::vector<Sample *> batch;
        batch.reserve(end - index);
        for (size_t i = index; i < end; ++i)
            batch.push_back(&dataset.samples[indices[i]]);
        index = end;
        return batch;
    }

    void reset()
    {
        index = 0;
        if (shuffle)
        {
            std::mt19937 rng(std::random_device{}());
            std::shuffle(indices.begin(), indices.end(), rng);
        }
    }
};
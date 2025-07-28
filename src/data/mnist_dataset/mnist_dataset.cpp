#include "mnist_dataset.h"

#include <algorithm> // for std::shuffle
#include <filesystem>
#include <random> // for std::default_random_engine

#define STB_IMAGE_IMPLEMENTATION
#include "../../../libs/stb_image.h"

namespace fs = std::filesystem;

std::vector<LabeledDataItem> MnistDataset::loadImages(std::string folder_path) {
  std::vector<LabeledDataItem> images;
  // loop through the mnist_png
  for (const auto &entry : fs::directory_iterator(folder_path)) {
    // loop through train/test
    for (const auto &entry : fs::directory_iterator(entry)) {
      // loop through label folders
      std::string label_directory = entry.path().filename().string();
      if (entry.is_directory()) {
        for (const auto &file : fs::directory_iterator(entry)) {
          if (file.is_regular_file()) {
            int width, height, channels;

            std::string image_filepath = file.path().string();

            unsigned char *tmp_image = stbi_load(image_filepath.c_str(), &width,
                                                 &height, &channels, 0);
            if (!tmp_image) {
              throw std::runtime_error("Failed to load image: " +
                                       image_filepath);
            }
            unsigned int label = std::stoul(label_directory);
            LabeledDataItem image =
                LabeledDataItem(tmp_image, width, height, channels, label);
            images.push_back(std::move(image));

            stbi_image_free(tmp_image);
          }
        }
      }
    }
  }
  return images;
}

void MnistDataset::shuffleDataset(std::vector<LabeledDataItem> &data) {
  // Shuffle the dataset
  std::random_device rd;  // Seed generator
  std::mt19937 gen(rd()); // Random number generator
  std::shuffle(data.begin(), data.end(), gen);
}

void MnistDataset::splitDataset(const std::vector<LabeledDataItem> &data,
                                int train_split, int val_split,
                                int test_split) {
  int number_of_train_images =
      data.size() * (static_cast<float>(train_split) / 100.0);
  int number_of_val_images =
      data.size() * (static_cast<float>(val_split) / 100.0);
  int number_of_test_images =
      data.size() * (static_cast<float>(test_split) / 100.0);

  // Ensure the resulting vectors are sized correctly
  train_data.reserve(number_of_train_images);
  val_data.reserve(number_of_val_images);
  test_data.reserve(number_of_test_images);

  // Distribute images into the respective vectors
  auto it = data.begin();
  train_data.insert(train_data.end(), it, it + number_of_train_images);
  it += number_of_train_images;

  val_data.insert(val_data.end(), it, it + number_of_val_images);
  it += number_of_val_images;

  test_data.insert(test_data.end(), it, data.end());
}

MnistDataset::MnistDataset(const std::string folder_path, int train_split,
                           int val_split, int test_split)
    : folder_path(folder_path) {
  // Validate that the splits sum up to 100
  if (train_split + val_split + test_split != 100) {
    throw std::runtime_error(
        "Error: Splits must sum to 100. Given: train_split = " +
        std::to_string(train_split) +
        ", val_split = " + std::to_string(val_split) +
        ", test_split = " + std::to_string(test_split));
  }
  std::vector<LabeledDataItem> images = loadImages(folder_path);
  shuffleDataset(images);
  splitDataset(images, train_split, val_split, test_split);
}

MnistDataset::~MnistDataset() {}

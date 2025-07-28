#include <gtest/gtest.h>

#include "../../src/data/mnist_dataset/mnist_dataset.h"

const std::string mnist_png_test_path =
    std::string(PROJECT_TESTS_DIR) + "/data/mnist_png_test";

TEST(MNIST, CREATE_DATASET) {

  MnistDataset mnist_dataset = MnistDataset(mnist_png_test_path, 80, 12, 8);
  int number_of_images_with_label_0 = 0;
  for (size_t i = 0; i < mnist_dataset.getSizeTrain(); ++i) {
    if (mnist_dataset.getItemTrain(i).label == 0)
      number_of_images_with_label_0++;
  }
  for (size_t i = 0; i < mnist_dataset.getSizeTest(); ++i) {
    if (mnist_dataset.getItemTest(i).label == 0)
      number_of_images_with_label_0++;
  }
  for (size_t i = 0; i < mnist_dataset.getSizeVal(); ++i) {
    if (mnist_dataset.getItemVal(i).label == 0)
      number_of_images_with_label_0++;
  }

  ASSERT_EQ(mnist_dataset.getSize(), 20);
  ASSERT_EQ(mnist_dataset.getSizeTrain(), 16);
  ASSERT_EQ(mnist_dataset.getSizeVal(), 2);
  ASSERT_EQ(mnist_dataset.getSizeTest(), 2);
  ASSERT_EQ(number_of_images_with_label_0, 2);
}

TEST(MNIST, CREATE_DATASET_SPLIT_ERROR) {
  // Test for invalid splits
  EXPECT_THROW(MnistDataset mnist_dataset =
                   MnistDataset(mnist_png_test_path, 80, 5, 5),
               std::runtime_error);
  EXPECT_THROW(MnistDataset mnist_dataset =
                   MnistDataset(mnist_png_test_path, 60, 30, 20),
               std::runtime_error);
  EXPECT_THROW(MnistDataset mnist_dataset =
                   MnistDataset(mnist_png_test_path, 50, 50, 10),
               std::runtime_error);
}

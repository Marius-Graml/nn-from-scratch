#include <gtest/gtest.h>

#include "../../libs/stb_image.h"
#include "../../src/data/labeled_data_item.h"

TEST(LABELEDDATAITEM, LABELEDDATAITEM_FROM_STB) {
  std::string image_path =
      std::string(PROJECT_TESTS_DIR) + "/data/mnist_png_test/train/0/1.png";

  int width, height, channels;
  unsigned char *tmp_image =
      stbi_load(image_path.c_str(), &width, &height, &channels, 0);
  LabeledDataItem image =
      LabeledDataItem(tmp_image, width, height, channels, 0);

  EXPECT_EQ(image.data_item[126], 0);
  EXPECT_EQ(image.data_item[127], 51);
  EXPECT_EQ(image.data_item[128], 159);
  EXPECT_EQ(image.data_item[129], 253);
  EXPECT_EQ(image.label, 0);

  stbi_image_free(tmp_image);
}

#ifndef LABELEDDATAITEM_H
#define LABELEDDATAITEM_H

#include "../../libs/Eigen/Dense"

class LabeledDataItem {
public:
  Eigen::VectorXd data_item;
  unsigned int label;

  // Constructor to initialize the attributes
  LabeledDataItem(const Eigen::VectorXd &data_item, unsigned int label)
      : data_item(data_item), label(label) {}

  // Constructor to initialize from raw pixel data
  LabeledDataItem(const unsigned char *stbi_image, unsigned int width,
                  unsigned int height, unsigned int channels,
                  unsigned int label)
      : LabeledDataItem(Eigen::VectorXd(width * height * channels), label) {
    int number_of_pixels = width * height * channels;

    for (int i = 0; i < number_of_pixels; ++i) {
      this->data_item[i] = static_cast<double>(stbi_image[i]);
    }
  }
};

#endif
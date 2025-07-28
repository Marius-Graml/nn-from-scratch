#ifndef DATASET_H
#define DATASET_H

#include "labeled_data_item.h"
#include <cstddef>
#include <vector>

class Dataset {
private:
  virtual void splitDataset(const std::vector<LabeledDataItem> &data,
                            int train_split, int val_split, int test_split) = 0;
  virtual void shuffleDataset(std::vector<LabeledDataItem> &data) = 0;

protected:
  std::vector<LabeledDataItem> train_data;
  std::vector<LabeledDataItem> val_data;
  std::vector<LabeledDataItem> test_data;

public:
  virtual ~Dataset() {}

  // Default implementation for getting the size of the dataset
  virtual size_t getSize() const {
    return train_data.size() + val_data.size() + test_data.size();
  }

  // Default implementation for training data
  virtual LabeledDataItem getItemTrain(size_t index) const {
    if (index >= train_data.size()) {
      throw std::out_of_range("Index out of range for training data.");
    }
    return train_data[index];
  }

  virtual size_t getSizeTrain() const { return train_data.size(); }

  // Default implementation for validation data
  virtual LabeledDataItem getItemVal(size_t index) const {
    if (index >= val_data.size()) {
      throw std::out_of_range("Index out of range for validation data.");
    }
    return val_data[index];
  }

  virtual size_t getSizeVal() const { return val_data.size(); }

  // Default implementation for test data
  virtual LabeledDataItem getItemTest(size_t index) const {
    if (index >= test_data.size()) {
      throw std::out_of_range("Index out of range for test data.");
    }
    return test_data[index];
  }

  virtual size_t getSizeTest() const { return test_data.size(); }
};

#endif
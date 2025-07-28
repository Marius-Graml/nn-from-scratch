#ifndef DUMMYDATASET_H
#define DUMMYDATASET_H

#include "../../src/data/dataset.h"

class DummyDataset : public Dataset {
private:
  void splitDataset(const std::vector<LabeledDataItem> &data,
                    [[maybe_unused]] int train_split,
                    [[maybe_unused]] int val_split,
                    [[maybe_unused]] int test_split) {
    for (size_t i = 0; i < 8; ++i) {
      train_data.push_back(data[i]);
    }

    // Populate val_data
    for (size_t i = 8; i < 11; ++i) {
      val_data.push_back(data[i]);
    }
  }
  void shuffleDataset([[maybe_unused]] std::vector<LabeledDataItem> &data) {}

public:
  DummyDataset(int data_size) {
    std::vector<LabeledDataItem> data = {
        LabeledDataItem(Eigen::VectorXd::Random(data_size), 0),
        LabeledDataItem(Eigen::VectorXd::Random(data_size), 1),
        LabeledDataItem(Eigen::VectorXd::Random(data_size), 2),
        LabeledDataItem(Eigen::VectorXd::Random(data_size), 3),
        LabeledDataItem(Eigen::VectorXd::Random(data_size), 0),
        LabeledDataItem(Eigen::VectorXd::Random(data_size), 1),
        LabeledDataItem(Eigen::VectorXd::Random(data_size), 2),
        LabeledDataItem(Eigen::VectorXd::Random(data_size), 3),
        LabeledDataItem(Eigen::VectorXd::Random(data_size), 3),
        LabeledDataItem(Eigen::VectorXd::Random(data_size), 0),
        LabeledDataItem(Eigen::VectorXd::Random(data_size), 1),
    };

    splitDataset(data, 80, 20, 0);
  }
};

#endif
#ifndef MNISTDATASET_H
#define MNISTDATASET_H

#include "../dataset.h"

class MnistDataset : public Dataset {
private:
  const std::string folder_path;

  std::vector<LabeledDataItem> loadImages(std::string folder_path);

  void shuffleDataset(std::vector<LabeledDataItem> &data) override;

  void splitDataset(const std::vector<LabeledDataItem> &data, int train_split,
                    int val_split, int test_split) override;

public:
  MnistDataset(const std::string folder_path, int train_split, int val_split,
               int test_split);
  ~MnistDataset();
};

#endif
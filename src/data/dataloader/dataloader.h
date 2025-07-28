#ifndef DATALOADER_H
#define DATALOADER_H

#include "../batch.h"
#include "../dataset.h"
#include "../labeled_data_item.h"
#include <vector>

class Dataloader {
private:
  const Dataset &dataset;
  int batch_size;

  std::vector<Batch>
  createBatches(std::function<size_t()> getSize,
                std::function<LabeledDataItem(size_t)> getItem);

public:
  Dataloader(const Dataset &dataset, int batch_size)
      : dataset(dataset), batch_size(batch_size) {}

  std::vector<Batch> getTrainBatches();

  std::vector<Batch> getValBatches();
};

#endif
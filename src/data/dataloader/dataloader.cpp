#include "dataloader.h"

#include "../../../libs/Eigen/Dense"

std::vector<Batch> Dataloader::createBatches(
    std::function<size_t()> getSize, // Callable for getSize
    std::function<LabeledDataItem(size_t)> getItem) {
  std::vector<Batch> batches;

  int total_data_items = getSize(); // Total training samples
  int num_batches = static_cast<int>(std::ceil(
      getSize() / static_cast<float>(batch_size))); // Total number of batches

  // Iterate over each batch
  for (int batch_index = 0; batch_index < num_batches; ++batch_index) {
    int start_index = batch_index * batch_size;
    int end_index = std::min(start_index + batch_size, total_data_items);

    // Create matrices for this batch
    int current_batch_size = end_index - start_index;
    Eigen::MatrixXd data_items(current_batch_size, getItem(0).data_item.size());
    Eigen::MatrixXd labels(current_batch_size, 1);

    // Fill the matrices with data
    for (int item_index = start_index; item_index < end_index; ++item_index) {
      int localIndex = item_index - start_index;
      data_items.row(localIndex) = getItem(item_index).data_item;
      labels(localIndex, 0) = static_cast<double>(getItem(item_index).label);
    }

    // Add batch to the list
    batches.push_back(Batch(data_items, labels));
  }

  return batches;
}

std::vector<Batch> Dataloader::getTrainBatches() {
  return createBatches(
      [&]() { return dataset.getSizeTrain(); },
      [&](size_t index) { return dataset.getItemTrain(index); });
}

std::vector<Batch> Dataloader::getValBatches() {
  return createBatches([&]() { return dataset.getSizeVal(); },
                       [&](size_t index) { return dataset.getItemVal(index); });
}
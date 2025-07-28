#include <gtest/gtest.h>
#include <iostream>

#include "../../src/data/dataloader/dataloader.h"
#include "dummy_dataset.h"

TEST(DATALOADER, CREATE_BATCH) {
  int data_size = 7;
  DummyDataset dummy_dataset = DummyDataset(data_size);

  int batch_size = 2;
  Dataloader dataloader = Dataloader(dummy_dataset, batch_size);
  std::vector<Batch> train_batches = dataloader.getTrainBatches();
  std::vector<Batch> val_batches = dataloader.getValBatches();

  ASSERT_EQ(train_batches.size(), 4);
  ASSERT_EQ(val_batches.size(), 2);
  ASSERT_EQ(train_batches[0].data.rows(), batch_size);
  ASSERT_EQ(train_batches[0].data.cols(), data_size);
  ASSERT_EQ(train_batches[0].labels.rows(), batch_size);
  ASSERT_EQ(train_batches[0].labels.cols(), 1);

  std::cout << train_batches[0].labels;
  std::cout << std::endl;
}
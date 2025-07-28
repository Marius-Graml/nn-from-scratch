#include "src/data/batch.h"
#include "src/data/dataloader/dataloader.h"
#include "src/data/mnist_dataset/mnist_dataset.h"
#include "src/network/losses/loss.h"
#include "src/network/losses/softmax_cross_entropy.h"
#include "src/network/network.h"
#include "src/network/optimizer/optimizer.h"
#include "src/network/optimizer/sgd.h"
#include "src/solver.h"
#include <iostream>
#include <memory>
#include <vector>

int main()
{
    std::string mnist_png_path = "mnist_png";
    int batch_size = 32;
    std::cout << "Init Dataloader" << std::endl;
    MnistDataset mnist_dataset = MnistDataset(mnist_png_path, 80, 10, 10);
    Dataloader dataloader = Dataloader(mnist_dataset, batch_size);

    double learning_rate = 0.002;

    std::cout << "Init Network" << std::endl;
    std::shared_ptr<Network> network =
        std::make_shared<Network>(Network(2, 28 * 28, 64, 10, 0));
    std::shared_ptr<Loss> loss_func =
        std::make_shared<SoftmaxCrossEntropyLoss>(SoftmaxCrossEntropyLoss());
    std::shared_ptr<Optimizer> optimizer =
        std::make_shared<SGD>(SGD(network, loss_func, learning_rate));

    std::cout << "Load Data" << std::endl;
    std::shared_ptr<std::vector<Batch>> train_batches =
        std::make_shared<std::vector<Batch>>(dataloader.getTrainBatches());
    std::shared_ptr<std::vector<Batch>> val_batches =
        std::make_shared<std::vector<Batch>>(dataloader.getValBatches());

    std::cout << "Start Training" << std::endl;
    Solver solver =
        Solver(network, train_batches, val_batches, loss_func, optimizer);
    solver.train(10);
    double train_accuracy = solver.get_dataset_accuracy(train_batches);
    std::cout << "Train Accuracy: " << train_accuracy << std::endl;
    double val_accuracy = solver.get_dataset_accuracy(val_batches);
    std::cout << "Val Accuracy: " << val_accuracy << std::endl;
    return 0;
}
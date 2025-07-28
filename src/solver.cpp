#include "solver.h"
#include <cmath>
#include <iostream>
#include <limits>

void Solver::reset() {
  train_loss_history.clear();
  val_loss_history.clear();
  train_batch_loss.clear();
  val_batch_loss.clear();

  current_patience = 0;
  best_model_stats["val_loss"] = std::numeric_limits<double>::max();
  best_model_stats["train_loss"] = std::numeric_limits<double>::max();
  best_params.clear();
}

double Solver::step(Eigen::MatrixXd X, Eigen::MatrixXd y, bool validation) {
  double loss = 0;

  auto y_pred = network->forward(X);

  Eigen::MatrixXd y_truth = y.transpose();
  loss = loss_func->forward(y_pred, y_truth);

  for (const auto &pair : network->regs) {
    loss += pair.second;
  }

  if (!validation) {
    optimizer->backward(y_pred, y_truth);
    optimizer->step();
  }

  return loss;
}

void Solver::update_best_loss(double train_loss, double val_loss) {
  if (val_loss < best_model_stats["val_loss"]) {
    best_model_stats["val_loss"] = val_loss;
    best_model_stats["train_loss"] = train_loss;
    best_params = network->params;
    current_patience = 0;
  } else {
    current_patience += 1;
  }
}

Solver::Solver(std::shared_ptr<Network> network,
               std::shared_ptr<std::vector<Batch>> train_batches,
               std::shared_ptr<std::vector<Batch>> val_batches,
               std::shared_ptr<Loss> loss_func,
               std::shared_ptr<Optimizer> optimizer)
    : network(network), train_batches(train_batches), val_batches(val_batches),
      loss_func(loss_func), optimizer(optimizer) {
  reset();
}

void Solver::train(int epochs) {
  for (int epoch = 0; epoch < epochs; epoch++) {
    double train_epoch_loss = 0.0;

    // Training
    std::cout << "Epoch " << epoch + 1 << ": Train..." << std::endl;
    for (const auto &batch : *train_batches) {
      auto X = batch.data;
      auto y = batch.labels;

      bool validation = false;
      if (epoch == 0) {
        validation = true;
      }
      auto train_loss = step(X, y, validation);

      train_batch_loss.push_back(train_loss);
      train_epoch_loss += train_loss;
    }

    train_epoch_loss /= train_batches->size();

    optimizer->learning_rate *= lr_decay;

    // Validation
    double val_epoch_loss = 0.0;

    std::cout << "Epoch " << epoch + 1 << ": Validate..." << std::endl;
    for (const auto &batch : *val_batches) {
      auto X = batch.data;
      auto y = batch.labels;

      auto val_loss = step(X, y, true);
      val_batch_loss.push_back(val_loss);
      val_epoch_loss += val_loss;
    }

    train_loss_history.push_back(train_epoch_loss);
    val_loss_history.push_back(val_epoch_loss);

    std::cout << "Epoch " << epoch + 1 << "/" << epochs
              << "; train loss: " << train_epoch_loss
              << "; val loss: " << val_epoch_loss << std::endl;

    update_best_loss(train_epoch_loss, val_epoch_loss);
  }

  network->params = best_params;
}

double
Solver::get_dataset_accuracy(std::shared_ptr<std::vector<Batch>> batches) {
  int correct = 0;
  double total = 0.0;

  for (const auto &batch : *batches) {
    auto X = batch.data;
    auto y = batch.labels;

    auto y_pred = network->forward(X);
    Eigen::MatrixXd indices(y_pred.rows(), 1);

    for (int i = 0; i < y_pred.rows(); ++i) {
      int maxIndex;
      y_pred.row(i).maxCoeff(
          &maxIndex); // Get the index of the max column in row i
      indices(i, 0) = maxIndex;
    }
    for (int i = 0; i < y_pred.rows(); ++i) {
      total = total + 1.0;
      auto predicted_label = static_cast<int>(indices(i, 0));
      auto true_label = static_cast<int>(y(i, 0));
      if (predicted_label == true_label)
        correct++;
    }
  }
  return correct / total;
}

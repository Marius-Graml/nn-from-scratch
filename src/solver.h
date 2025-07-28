#ifndef SOLVER_H
#define SOLVER_H

#include "data/batch.h"
#include "network/losses/loss.h"
#include "network/network.h"
#include "network/optimizer/optimizer.h"
#include <vector>

class Solver {
private:
  std::shared_ptr<Network> network;
  std::shared_ptr<std::vector<Batch>> train_batches;
  std::shared_ptr<std::vector<Batch>> val_batches;
  std::shared_ptr<Loss> loss_func;
  std::shared_ptr<Optimizer> optimizer;

  int current_patience;
  std::vector<double> train_loss_history;
  std::vector<double> val_loss_history;
  std::vector<double> train_batch_loss;
  std::vector<double> val_batch_loss;
  std::unordered_map<std::string, double> best_model_stats;
  std::unordered_map<std::string, Eigen::MatrixXd> best_params;

  double lr_decay = 1.0;

  void reset();
  void update_best_loss(double train_loss, double val_loss);
  double step(Eigen::MatrixXd X, Eigen::MatrixXd y, bool validation);

public:
  Solver(std::shared_ptr<Network> network,
         std::shared_ptr<std::vector<Batch>> train_batches,
         std::shared_ptr<std::vector<Batch>> val_batches,
         std::shared_ptr<Loss> loss_func, std::shared_ptr<Optimizer> optimizer);

  void train(int epochs);
  double get_dataset_accuracy(std::shared_ptr<std::vector<Batch>> batches);
};

#endif
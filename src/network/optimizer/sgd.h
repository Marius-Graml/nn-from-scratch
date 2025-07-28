#ifndef SGD_H
#define SGD_H

#include "../../../libs/Eigen/Dense"
#include "../losses/softmax_cross_entropy.h"
#include "../network.h"
#include "optimizer.h"
#include <memory>

class SGD : public Optimizer {
private:
  std::shared_ptr<Network> network;
  std::shared_ptr<Loss> loss_func;

  Eigen::MatrixXd update(Eigen::MatrixXd w, Eigen::MatrixXd dw);

public:
  double learning_rate;
  SGD(std::shared_ptr<Network> network, std::shared_ptr<Loss> loss_func,
      double lr)
      : network(network), loss_func(loss_func), learning_rate(lr) {}
  void backward(Eigen::MatrixXd y_pred, Eigen::MatrixXd y_true) override;
  void step() override;
};

#endif

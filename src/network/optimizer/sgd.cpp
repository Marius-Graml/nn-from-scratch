#include "sgd.h"
#include <iostream>

Eigen::MatrixXd SGD::update(Eigen::MatrixXd w, Eigen::MatrixXd dw) {
  w = w - learning_rate * dw;
  return w;
}

void SGD::backward(Eigen::MatrixXd y_pred, Eigen::MatrixXd y_true) {
  auto dout = loss_func->backward(y_pred, y_true);
  network->backward(dout);
}

void SGD::step() {
  for (const auto &[key, value] : network->gradients) {
    auto w = network->params[key];
    auto dw = network->gradients[key];

    if (key.find('b') != std::string::npos) {
      dw = dw.transpose();
    }

    auto w_updated = update(w, dw);
    network->params[key] = w_updated;
    network->gradients[key] = Eigen::MatrixXd::Zero(dw.rows(), dw.cols());
  }
}

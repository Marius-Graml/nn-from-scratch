#include "mse.h"

double MSELoss::forward(const Eigen::MatrixXd &y_out,
                        const Eigen::MatrixXd &y_truth) {

  auto loss = (y_out - y_truth).array().square().mean();

  return loss;
}

Eigen::MatrixXd MSELoss::backward(const Eigen::MatrixXd &y_out,
                                  const Eigen::MatrixXd &y_truth) {
  Eigen::MatrixXd gradient = (2.0 / y_out.size()) * (y_out - y_truth);
  return gradient;
}
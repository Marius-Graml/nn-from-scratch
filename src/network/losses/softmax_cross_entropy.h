#ifndef SOFTMAX_CROSS_ENTROPY_H
#define SOFTMAX_CROSS_ENTROPY_H

#include "../../../libs/Eigen/Dense"
#include "loss.h"
#include <cmath>

class SoftmaxCrossEntropyLoss : public Loss {
public:
  Eigen::MatrixXd y_out_probs_cache;

  double forward(const Eigen::MatrixXd &y_out,
                 const Eigen::MatrixXd &y_truth) override;
  Eigen::MatrixXd backward(const Eigen::MatrixXd &y_out,
                           const Eigen::MatrixXd &y_truth) override;
};

#endif

#ifndef MSE_H
#define MSE_H

#include "../../../libs/Eigen/Dense"
#include "loss.h"

class MSELoss : public Loss {
public:
  double forward(const Eigen::MatrixXd &y_out,
                 const Eigen::MatrixXd &y_truth) override;

  Eigen::MatrixXd backward(const Eigen::MatrixXd &y_out,
                           const Eigen::MatrixXd &y_truth) override;
};

#endif

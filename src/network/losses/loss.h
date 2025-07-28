#ifndef LOSS_H
#define LOSS_H

#include "../../../libs/Eigen/Dense"

class Loss {
public:
  virtual ~Loss() = default;
  virtual double forward(const Eigen::MatrixXd &y_out,
                         const Eigen::MatrixXd &y_truth) = 0;
  virtual Eigen::MatrixXd backward(const Eigen::MatrixXd &y_out,
                                   const Eigen::MatrixXd &y_truth) = 0;
};

#endif
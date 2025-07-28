#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../layers/layer.h"
#include <memory>
#include <vector>

class Optimizer {
public:
  double learning_rate;
  virtual ~Optimizer() {}
  virtual void step() = 0;
  virtual void backward(Eigen::MatrixXd y_pred, Eigen::MatrixXd y_true) = 0;
};

#endif

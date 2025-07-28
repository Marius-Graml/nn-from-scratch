/*#ifndef ADAM_H
#define ADAM_H

#include "../../../libs/Eigen/Dense"
#include "../layers/linear.h"
#include "optimizer.h"
#include <iostream>
#include <stdexcept>

class Adam : public Optimizer {
private:
  double learning_rate;
  double beta_1;
  double beta_2;
  double epsilon;
  int iteration = 1;
  std::unordered_map<Layer *, Eigen::MatrixXd> m_W, v_W;
  std::unordered_map<Layer *, Eigen::VectorXd> m_b, v_b;

public:
  Adam(double lr, double b_1, double b_2, double e)
      : learning_rate(lr), beta_1(b_1), beta_2(b_2), epsilon(e) {}
  void step(std::vector<std::shared_ptr<Layer>> &layers) override;
};

#endif*/

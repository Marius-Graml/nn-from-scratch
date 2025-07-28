#ifndef LEAKY_RELU_H
#define LEAKY_RELU_H

#include "../../../libs/Eigen/Dense"
#include "layer.h"

class LeakyRelu : public Layer {
private:
  double alpha; // Slope for negative inputs

public:
  LeakyRelu(double alpha = 0.01) : alpha(alpha) {} // Default alpha = 0.01

  std::vector<Eigen::MatrixXd> forward(Eigen::MatrixXd Y) override {
    // Apply Leaky ReLU: Y if Y > 0, alpha * Y otherwise
    Eigen::MatrixXd Z = (Y.array() * (Y.array() > 0).cast<double>() +
                         alpha * Y.array() * (Y.array() <= 0).cast<double>())
                            .matrix();
    std::vector<Eigen::MatrixXd> cache = {Z, Y};
    return cache;
  }

  Eigen::MatrixXd backward(Eigen::MatrixXd dout,
                           std::vector<Eigen::MatrixXd> cache) override {
    // Gradient is 1 for positive inputs, alpha for negative inputs
    Eigen::MatrixXd dX;
    auto Z = cache[0];
    dX = (dout.array() * ((Z.array() > 0).cast<double>() +
                          alpha * (Z.array() <= 0).cast<double>()))
             .matrix();
    return dX;
  }
};

#endif

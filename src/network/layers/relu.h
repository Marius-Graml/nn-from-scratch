#ifndef RELU_H
#define RELU_H

#include "../../../libs/Eigen/Dense"
#include "layer.h"

class Relu : public Layer {
public:
  Relu() {}

  std::vector<Eigen::MatrixXd> forward(Eigen::MatrixXd Y) override {

    Eigen::MatrixXd Z = (Y.array() * (Y.array() > 0).cast<double>()).matrix();

    std::vector<Eigen::MatrixXd> cache = {Z, Y};
    return cache;
    ;
  }

  Eigen::MatrixXd backward(Eigen::MatrixXd dout,
                           std::vector<Eigen::MatrixXd> cache) override {

    Eigen::MatrixXd dX;
    auto Z = cache[0];

    dX = (dout.array() * (Z.array() > 0).cast<double>()).matrix();

    return dX;
  }
};

#endif

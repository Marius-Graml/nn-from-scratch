
#ifndef SIGMOID_H
#define SIGMOID_H

#include "../../../libs/Eigen/Dense"
#include "layer.h"
#include <iostream>

class Sigmoid : public Layer {
public:
  Sigmoid() {}

  std::vector<Eigen::MatrixXd> forward(Eigen::MatrixXd Y) override {
    Eigen::MatrixXd Z = (1.0 / (1.0 + (-Y.array()).exp())).matrix();

    std::vector<Eigen::MatrixXd> cache = {Z, Y};
    return cache;
  }

  Eigen::MatrixXd backward(Eigen::MatrixXd dout,
                           std::vector<Eigen::MatrixXd> cache) override {
    Eigen::MatrixXd dX;
    auto Z = cache[0];
    std::cout << "[DEBUG] Sigmoid Forward Output Shape: " << Z.rows() << " x "
              << Z.cols() << std::endl;

    dX = (dout.array() * Z.array() * (1 - Z.array())).matrix();

    return dX;
  }
};

#endif
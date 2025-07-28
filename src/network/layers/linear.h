
#ifndef LINEAR_H
#define LINEAR_H

#include "../../../libs/Eigen/Dense"
#include <iostream>
#include <stdexcept>

class Linear {
public:
  static std::vector<Eigen::MatrixXd>
  forward(Eigen::MatrixXd X, Eigen::MatrixXd W, Eigen::VectorXd b) {
    // X: Input data of shape (N, d_1, ... d_k)
    Eigen::MatrixXd Y;

    if (X.cols() != W.rows()) {
      throw(std::runtime_error("Dimensions of X and W do not match!"));
    }
    if (W.cols() != b.size()) {
      throw(std::invalid_argument("Dimensions of W and b do not match!"));
    }
    Y = (X * W).rowwise() + b.transpose();
    std::vector<Eigen::MatrixXd> cache = {Y, X, W};
    return cache;
  }

  static std::vector<Eigen::MatrixXd>
  backward(Eigen::MatrixXd dout, std::vector<Eigen::MatrixXd> cache) {
    // dout: Upstream gradient of shape (N,M)
    // Struct of:
    // X: Input data of shape (N, d_1, ... d_k)

    auto X = cache[1];
    auto W = cache[2];

    auto dW = X.transpose() * dout;
    auto dX = dout * W.transpose();
    auto db = dout.colwise().sum();

    // Debugging Outputs
    /*std::cout << "[DEBUG] Backward pass in Linear Layer\n";
    std::cout << "[DEBUG] dW shape: " << dW.rows() << " x " << dW.cols()
              << std::endl;
    if (dW.rows() > 0) {
      std::cout << "[DEBUG] dW (first few rows): \n"
                << dW.topRows(std::min(5, static_cast<int>(dW.rows())))
                << std::endl;
    }
    std::cout << "[DEBUG] db shape: " << db.rows() << " x " << db.cols()
              << std::endl;
    std::cout << "[DEBUG] db: " << db.transpose() << std::endl;
    std::cout << "[DEBUG] dW min/max: " << dW.minCoeff() << " / "
              << dW.maxCoeff() << std::endl;
    std::cout << "[DEBUG] db min/max: " << db.minCoeff() << " / "
              << db.maxCoeff() << std::endl;

*/
    std::vector<Eigen::MatrixXd> d_cache = {dX, dW, db};
    return d_cache;
  }
};

#endif
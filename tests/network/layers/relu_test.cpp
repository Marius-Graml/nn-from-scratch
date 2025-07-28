#include <gtest/gtest.h>

#include "../../../src/network/layers/relu.h"

#include "../../../libs/Eigen/Dense"
#include "round.h"

TEST(relu, forward) {
  // Initialize inputs
  Eigen::MatrixXd X(3, 3);
  X << -1, 0.4, 1.1, 0.4, -0.2, 1, 0.1, -0.5, 1;

  // Define layers
  Relu relu = Relu();

  // Forward pass
  auto cache = relu.forward(X);
  Eigen::MatrixXd Zr = cache[0];

  // Round
  Eigen::MatrixXd Zr_r = roundMatrixToFiveDecimalPlaces(Zr);

  // Define correct results, calculated in MATLAB
  // ReLU Output Matrix
  Eigen::MatrixXd Zr_c(3, 3);
  Zr_c << 0.00000, 0.40000, 1.10000, 0.40000, 0.00000, 1.00000, 0.10000,
      0.00000, 1.00000;

  // Compare results
  EXPECT_EQ(Zr_r, Zr_c); // Test relu forward
}

TEST(relu, backward) {
  // Initialize inputs
  Eigen::MatrixXd X(3, 3);
  X << -1, 0.4, 1.1, 0.4, -0.2, 1, 0.1, -0.5, 1;

  Eigen::MatrixXd dout(3, 3);
  dout << 3, 3, 4, 5, 2, 6, 7, 6, 7;

  // Define layers
  Relu relu = Relu();

  // Forward pass
  auto cache = relu.forward(X);
  Eigen::MatrixXd Zr = cache[0];

  // Backward pass
  Eigen::MatrixXd dX_relu = relu.backward(dout, cache);

  // Round
  Eigen::MatrixXd dX_relu_r = roundMatrixToFiveDecimalPlaces(dX_relu);

  // Define correct results, calculated in MATLAB
  // dX_relu Matrix
  Eigen::MatrixXd dX_relu_c(3, 3);
  dX_relu_c << 0.00000, 3.00000, 4.00000, 5.00000, 0.00000, 6.00000, 7.00000,
      0.00000, 7.00000;

  // Compare results
  EXPECT_EQ(dX_relu_r, dX_relu_c); // Test relu backward
}

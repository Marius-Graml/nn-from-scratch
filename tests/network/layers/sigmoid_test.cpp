#include <gtest/gtest.h>

#include "../../../libs/Eigen/Dense"
#include "../../../src/network/layers/sigmoid.h"
#include "round.h"

TEST(sigmoid, forward) {
  // Initialize inputs and weights
  Eigen::MatrixXd X(3, 3);
  X << -1, 0.4, 1.1, 0.4, -0.2, 1, 0.1, -0.5, 1;

  // Define layers
  Sigmoid sigmoid = Sigmoid();

  // Forward pass
  auto cache = sigmoid.forward(X);
  Eigen::MatrixXd Zs = cache[0];

  // Round
  Eigen::MatrixXd Zs_r = roundMatrixToFiveDecimalPlaces(Zs);

  // Define correct results, calculated in MATLAB
  // Sigmoid Output Matrix
  Eigen::MatrixXd Zs_c(3, 3);
  Zs_c << 0.26894, 0.59869, 0.75026, 0.59869, 0.45017, 0.73106, 0.52498,
      0.37754, 0.73106;

  // Compare results
  EXPECT_EQ(Zs_r, Zs_c); // Test sigmoid forward
}

TEST(sigmoid, backward) {
  // Initialize inputs and weights
  Eigen::MatrixXd X(3, 3);
  X << -1, 0.4, 1.1, 0.4, -0.2, 1, 0.1, -0.5, 1;

  Eigen::MatrixXd dout(3, 3);
  dout << 3, 3, 4, 5, 2, 6, 7, 6, 7;

  // Define layers
  Sigmoid sigmoid = Sigmoid();

  // Forward pass
  auto cache = sigmoid.forward(X);
  Eigen::MatrixXd Zs = cache[0];

  // Backward pass
  Eigen::MatrixXd dX_sigmoid = sigmoid.backward(dout, cache);

  // Round
  Eigen::MatrixXd dX_sigmoid_r = roundMatrixToFiveDecimalPlaces(dX_sigmoid);

  // Define correct results, calculated in MATLAB
  // dX_sigmoid Matrix
  Eigen::MatrixXd dX_sigmoid_c(3, 3);
  dX_sigmoid_c << 0.58984, 0.72078, 0.74948, 1.20130, 0.49503, 1.17967, 1.74563,
      1.41002, 1.37628;

  // Compare results
  EXPECT_EQ(dX_sigmoid_r, dX_sigmoid_c); // Test sigmoid backward
}

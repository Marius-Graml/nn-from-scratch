#include <gtest/gtest.h>

#include "../../../libs/Eigen/Dense"
#include "../../../src/network/layers/linear.h"
#include "round.h"

TEST(linear, forward) {
  // Initialize inputs and weights
  Eigen::MatrixXd X(3, 3);
  X << -1, 0.4, 1.1, 0.4, -0.2, 1, 0.1, -0.5, 1;

  Eigen::MatrixXd W(3, 3);
  W << -1, 0.5, 0.1, 0.01, 0.9, 2, 0.4, 0.2, 0.3;

  Eigen::VectorXd b(3);
  b << 0.2, 0.2, 0.7;

  // Forward pass
  auto cache = Linear::forward(X, W, b);
  auto Y = cache[0];

  // Round to 5 decimal digits
  Eigen::MatrixXd Y_r = roundMatrixToFiveDecimalPlaces(Y);

  // Define correct results, calculated in MATLAB
  Eigen::MatrixXd Y_c(3, 3);
  Y_c << 1.64400, 0.28000, 1.73000, 0.19800, 0.42000, 0.64000, 0.49500, 0.00000,
      0.01000;

  // Compare results
  EXPECT_EQ(Y_r, Y_c); // Test linear forward
}

TEST(linear, backward) {
  // Initialize inputs and weights
  Eigen::MatrixXd X(3, 3);
  X << -1, 0.4, 1.1, 0.4, -0.2, 1, 0.1, -0.5, 1;

  Eigen::MatrixXd W(3, 3);
  W << -1, 0.5, 0.1, 0.01, 0.9, 2, 0.4, 0.2, 0.3;

  Eigen::VectorXd b(3);
  b << 0.2, 0.2, 0.7;

  Eigen::MatrixXd dout(3, 3);
  dout << 3, 3, 4, 5, 2, 6, 7, 6, 7;

  // Forward pass
  auto cache = Linear::forward(X, W, b);
  auto Y = cache[0];

  // Backward pass
  auto grads = Linear::backward(dout, cache);

  Eigen::MatrixXd dX = grads[0];
  Eigen::MatrixXd dW = grads[1];
  Eigen::MatrixXd db = grads[2];

  // Round to 5 decimal digits
  Eigen::MatrixXd Y_r = roundMatrixToFiveDecimalPlaces(Y);
  Eigen::MatrixXd dW_r = roundMatrixToFiveDecimalPlaces(dW);
  Eigen::MatrixXd dX_r = roundMatrixToFiveDecimalPlaces(dX);
  Eigen::MatrixXd db_r = roundMatrixToFiveDecimalPlaces(db);

  // Define correct results, calculated in MATLAB
  Eigen::MatrixXd Y_c(3, 3);
  Y_c << 1.64400, 0.28000, 1.73000, 0.19800, 0.42000, 0.64000, 0.49500, 0.00000,
      0.01000;

  // dW Matrix
  Eigen::MatrixXd dW_c(3, 3);
  dW_c << -0.30000, -1.60000, -0.90000, -3.30000, -2.20000, -3.10000, 15.30000,
      11.30000, 17.40000;

  // dX Matrix
  Eigen::MatrixXd dX_c(3, 3);
  dX_c << -1.10000, 10.73000, 3.00000, -3.40000, 13.85000, 4.20000, -3.30000,
      19.47000, 6.10000;

  // db Vector
  Eigen::MatrixXd db_c(1, 3);
  db_c << 15.00000, 11.00000, 17.00000;

  // Compare results
  EXPECT_EQ(Y_r, Y_c);   // Test linear forward
  EXPECT_EQ(dW_r, dW_c); // Test linear backward weights
  EXPECT_EQ(dX_r, dX_c); // Test linear backward input
  EXPECT_EQ(db_r, db_c); // Test linear backward biases
}

#include "../../../libs/Eigen/Dense"
#include "../../../src/network/losses/softmax_cross_entropy.h"
#include "round.h"
#include <gtest/gtest.h>

TEST(softmax_cross_entropy, forward) {
  Eigen::MatrixXd y_out(2, 3);
  y_out << 0.7, 2.3, -1.0, 0.4, 0.1, 1.0;

  Eigen::MatrixXd y_truth(1, 2);
  y_truth << 0, 1;

  // Use dynamic allocation to handle abstract class
  auto softmax_ce = SoftmaxCrossEntropyLoss();

  // Forward pass
  double actual_loss = softmax_ce.forward(y_out, y_truth);
  double expected_loss = 1.6923559905101797;

  Eigen::MatrixXd expected_y_out_probs_cache(2, 3);
  expected_y_out_probs_cache << 0.16298017, 0.80724604, 0.02977379, 0.28066732,
      0.20792347, 0.51140921;
  expected_y_out_probs_cache =
      roundMatrixToFiveDecimalPlaces(expected_y_out_probs_cache);
  auto actual_y_out_probs_cache =
      roundMatrixToFiveDecimalPlaces(softmax_ce.y_out_probs_cache);

  EXPECT_NEAR(expected_loss, actual_loss, 0.001);
  EXPECT_EQ(expected_y_out_probs_cache, actual_y_out_probs_cache);
}

TEST(softmax_cross_entropy, backward) {
  Eigen::MatrixXd y_out(2, 3);
  y_out << 0.7, 2.3, -1.0, 0.4, 0.1, 1.0;

  Eigen::MatrixXd y_truth(1, 2);
  y_truth << 0, 1;

  // Use dynamic allocation to handle abstract class
  auto softmax_ce = SoftmaxCrossEntropyLoss();

  // Forward pass
  softmax_ce.forward(y_out, y_truth);

  Eigen::MatrixXd expected_gradient(2, 3);
  expected_gradient << -0.41850992, 0.40362302, 0.0148869, 0.14033366,
      -0.39603827, 0.2557046;
  expected_gradient = roundMatrixToFiveDecimalPlaces(expected_gradient);
  auto actual_gradient =
      roundMatrixToFiveDecimalPlaces(softmax_ce.backward(y_out, y_truth));

  EXPECT_EQ(expected_gradient, actual_gradient);
}

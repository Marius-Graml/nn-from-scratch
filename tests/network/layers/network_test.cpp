#include "../../../libs/Eigen/Dense"
#include "../../../src/network/network.h"
#include <gtest/gtest.h>
#include <memory>

using AffineTuple = std::tuple<Eigen::MatrixXd, Eigen::MatrixXd,
                               Eigen::MatrixXd, Eigen::VectorXd>;
using ActivationTuple = std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>;

TEST(NetworkTest, InitWeights) {
  const int input_size = 3 * 32 * 32;
  const int hidden_size = 2;
  const int num_classes = 10;
  std::unique_ptr<Network> net =
      std::make_unique<Network>(4, input_size, hidden_size, num_classes, 0);

  std::unordered_map<std::string, Eigen::MatrixXd> initial_weights =
      net->get_params();

  // Check that we have the expected keys: W1-W4 and b1-b4
  for (int i = 1; i <= 4; ++i) {
    std::string w_key = "W" + std::to_string(i);
    std::string b_key = "b" + std::to_string(i);

    // Assert that both "W" and "b" keys exist in the map
    ASSERT_TRUE(initial_weights.find(w_key) != initial_weights.end())
        << "Key " << w_key << " not found!";
    ASSERT_TRUE(initial_weights.find(b_key) != initial_weights.end())
        << "Key " << b_key << " not found!";
  }

  // Check that W1 has the expected size (input_size, hidden_size)
  std::string w1_key = "W1";
  auto w1 = initial_weights[w1_key];

  ASSERT_EQ(w1.rows(), input_size);
  ASSERT_EQ(w1.cols(), hidden_size);

  std::string w2_key = "W2";
  auto w2 = initial_weights[w2_key];

  ASSERT_EQ(w2.rows(), hidden_size);
  ASSERT_EQ(w2.cols(), hidden_size);

  std::string w4_key = "W4";
  auto w4 = initial_weights[w4_key];

  ASSERT_EQ(w4.rows(), hidden_size);
  ASSERT_EQ(w4.cols(), num_classes);
}

TEST(NetworkTest, forward) {
  Eigen::MatrixXd X(2, 5);

  X << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;

  const int input_size = 5;
  const int hidden_size = 2;
  const int num_classes = 10;
  std::unique_ptr<Network> net =
      std::make_unique<Network>(4, input_size, hidden_size, num_classes, 0);

  net->set_params(2.5);
  Eigen::MatrixXd y_actual = net->forward(X);
  auto cache = net->get_cache();

  Eigen::MatrixXd y_expected(2, 10);
  y_expected << 5077.5, 5077.5, 5077.5, 5077.5, 5077.5, 5077.5, 5077.5, 5077.5,
      5077.5, 5077.5, 12890.0, 12890.0, 12890.0, 12890.0, 12890.0, 12890.0,
      12890.0, 12890.0, 12890.0, 12890.0;

  // Compare results
  EXPECT_EQ(y_expected, y_actual); // Test linear forward
}

TEST(NetworkTest, backward) {
  Eigen::MatrixXd X(2, 5);

  X << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;

  const int input_size = 5;
  const int hidden_size = 2;
  const int num_classes = 10;
  std::unique_ptr<Network> net =
      std::make_unique<Network>(4, input_size, hidden_size, num_classes, 0);

  net->set_params(2.5);
  Eigen::MatrixXd y = net->forward(X);
  std::unordered_map<std::string, Eigen::MatrixXd> gradients_actual =
      net->backward(y);
  auto w1_actual = gradients_actual["W1"];
  auto b1_actual = gradients_actual["b1"];
  /*for (const auto &[key, matrix] : gradients_actual) {
    std::cout << "Key: " << key << "\n";
    std::cout << "Matrix (" << matrix.rows() << "x" << matrix.cols() << "):\n";
    std::cout << matrix << "\n\n"; // Eigen supports streaming to std::cout
  }*/
  Eigen::MatrixXd w1_expected(5, 2);
  w1_expected << 51510937.5, 51510937.5, 62740625.0, 62740625.0, 73970312.5,
      73970312.5, 85200000.0, 85200000.0, 96429687.5, 96429687.5;
  Eigen::MatrixXd b1_expected(1, 2);
  b1_expected << 11229687.5, 11229687.5;

  // Compare results
  EXPECT_EQ(w1_expected, w1_actual);
  EXPECT_EQ(b1_expected, b1_actual);
}

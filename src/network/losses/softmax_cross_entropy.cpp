#include "softmax_cross_entropy.h"

// Forward pass: computes softmax probabilities and cross-entropy loss
double SoftmaxCrossEntropyLoss::forward(const Eigen::MatrixXd &y_out,
                                        const Eigen::MatrixXd &y_truth) {
  int N = y_out.rows();
  int C = y_out.cols();
  Eigen::MatrixXd y_truth_one_hot = Eigen::MatrixXd::Zero(N, C);
  for (int i = 0; i < N; ++i) {
    int class_idx = y_truth(0, i); // Extract class index from y_truth
    if (class_idx >= 0 && class_idx < C) {
      y_truth_one_hot(i, class_idx) = 1.0;
    }
  }

  // Apply softmax with numerical stability
  Eigen::MatrixXd y_out_exp =
      (y_out.array().colwise() - y_out.rowwise().maxCoeff().array()).exp();
  Eigen::MatrixXd y_out_probs =
      y_out_exp.array().colwise() / y_out_exp.rowwise().sum().array();

  // Compute cross-entropy loss
  Eigen::MatrixXd loss_matrix =
      -y_truth_one_hot.array() * y_out_probs.array().log();
  Eigen::VectorXd loss_vector = loss_matrix.rowwise().sum();

  double loss = loss_vector.mean();

  y_out_probs_cache = y_out_probs;

  return loss;
}

Eigen::MatrixXd
SoftmaxCrossEntropyLoss::backward(const Eigen::MatrixXd &y_out,
                                  const Eigen::MatrixXd &y_truth) {
  int N = y_out.rows();
  int C = y_out.cols();

  // Gradient calculation
  Eigen::MatrixXd gradient = y_out_probs_cache;

  Eigen::MatrixXd y_truth_one_hot = Eigen::MatrixXd::Zero(N, C);
  for (int i = 0; i < N; ++i) {
    int class_idx = y_truth(0, i); // Extract class index from y_truth
    if (class_idx >= 0 && class_idx < C) {
      y_truth_one_hot(i, class_idx) = 1.0;
    }
  }

  // Modify the gradient with respect to the true class
  for (int i = 0; i < N; ++i) {
    gradient.row(i) = gradient.row(i) - y_truth_one_hot.row(i);
  }

  gradient /= N;

  return gradient;
}
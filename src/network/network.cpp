#include "network.h"

#include "layers/linear.h"
#include <iostream>
#include <random>

Network::Network(int num_layer, int input_size, int hidden_size,
                 int num_classes, int reg_strength)
    : num_layer(num_layer), input_size(input_size), hidden_size(hidden_size),
      num_classes(num_classes), reg_strength(reg_strength) {
  init_weights();
  init_gradients();
  init_regs();
}

void Network::init_weights() {
  const double std = 0.001;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist(0.0, std); // Mean=0, Std=std_dev

  params["W1"] = Eigen::MatrixXd::NullaryExpr(input_size, hidden_size,
                                              [&]() { return dist(gen); });
  params["b1"] = Eigen::MatrixXd::Zero(hidden_size, 1);

  // Hidden layers
  for (int i = 0; i < num_layer - 2; ++i) {
    std::string w_key = "W" + std::to_string(i + 2);
    std::string b_key = "b" + std::to_string(i + 2);

    params[w_key] = Eigen::MatrixXd::NullaryExpr(hidden_size, hidden_size,
                                                 [&]() { return dist(gen); });
    params[b_key] = Eigen::MatrixXd::Zero(hidden_size, 1);
  }

  // Output layer
  std::string final_w_key = "W" + std::to_string(num_layer);
  std::string final_b_key = "b" + std::to_string(num_layer);

  params[final_w_key] = Eigen::MatrixXd::NullaryExpr(
      hidden_size, num_classes, [&]() { return dist(gen); });
  params[final_b_key] = Eigen::MatrixXd::Zero(num_classes, 1);
}

void Network::init_gradients() {
  gradients["W1"] = Eigen::MatrixXd::Zero(input_size, hidden_size);
  gradients["b1"] = Eigen::MatrixXd::Zero(hidden_size, 1);
  for (int i = 0; i < num_layer - 2; ++i) {
    std::string w_key = "W" + std::to_string(i + 2);
    std::string b_key = "b" + std::to_string(i + 2);

    gradients[w_key] = Eigen::MatrixXd::Zero(hidden_size, hidden_size);
    gradients[b_key] = Eigen::MatrixXd::Zero(hidden_size, 1);
  }
  gradients["W1"] = Eigen::MatrixXd::Zero(hidden_size, num_classes);
  gradients["b1"] = Eigen::MatrixXd::Zero(num_classes, 1);
}

void Network::init_regs() {
  for (int i = 0; i < num_layer; ++i) {
    regs["W" + std::to_string(i + 1)] = 0.0;
  }
}

void Network::set_params(double value) {
  for (auto &[key, matrix] : params) {
    matrix.setConstant(value); // Set all elements to 2.5
  }
}

Eigen::MatrixXd Network::forward(const Eigen::MatrixXd &X) {
  global_cache.clear();
  regs.clear();

  Eigen::MatrixXd X_updated = X;
  for (int i = 0; i < num_layer - 1; ++i) {
    Eigen::MatrixXd W = params["W" + std::to_string(i + 1)];
    Eigen::MatrixXd b = params["b" + std::to_string(i + 1)];

    auto cache_affine = Linear::forward(X_updated, W, b);
    X_updated = cache_affine[0];
    global_cache["affine" + std::to_string(i + 1)] = cache_affine;

    auto cache_activation = activation_function->forward(X_updated);
    X_updated = cache_activation[0];
    global_cache["activation" + std::to_string(i + 1)] = cache_activation;

    regs["W" + std::to_string(i + 1)] =
        (W.array().square().sum()) * reg_strength;
  }
  Eigen::MatrixXd W = params["W" + std::to_string(num_layer)];
  Eigen::MatrixXd b = params["b" + std::to_string(num_layer)];

  auto cache_affine = Linear::forward(X_updated, W, b);
  global_cache["affine" + std::to_string(num_layer)] = cache_affine;
  regs["W" + std::to_string(num_layer)] =
      (W.array().square().sum()) * reg_strength;

  return cache_affine[0];
}

std::unordered_map<std::string, Eigen::MatrixXd>
Network::backward(const Eigen::MatrixXd &dy) {
  auto cache_affine = global_cache["affine" + std::to_string(num_layer)];
  auto cache_affine_backward = Linear::backward(dy, cache_affine);

  Eigen::MatrixXd dX = cache_affine_backward[0];
  Eigen::MatrixXd dW = cache_affine_backward[1];
  Eigen::MatrixXd db = cache_affine_backward[2];

  gradients["W" + std::to_string(num_layer)] =
      dW + 2 * reg_strength * params["W" + std::to_string(num_layer)];
  gradients["b" + std::to_string(num_layer)] = db;

  for (int i = num_layer - 2; i >= 0; --i) {
    auto cache_activation = global_cache["activation" + std::to_string(i + 1)];
    auto cache_affine = global_cache["affine" + std::to_string(i + 1)];

    dX = activation_function->backward(dX, cache_activation);

    auto tmp_cache_affine_backward = Linear::backward(dX, cache_affine);

    dX = tmp_cache_affine_backward[0];
    dW = tmp_cache_affine_backward[1];
    db = tmp_cache_affine_backward[2];

    gradients["W" + std::to_string(i + 1)] =
        dW + 2 * reg_strength * params["W" + std::to_string(i + 1)];
    gradients["b" + std::to_string(i + 1)] = db;
  }

  return gradients;
}

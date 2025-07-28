#ifndef NETWORK_H
#define NETWORK_H

#include "../../libs/Eigen/Dense"
#include "layers/layer.h"
#include "layers/relu.h"
#include <memory>
#include <vector>

class Network {
private:
  std::unordered_map<std::string, std::vector<Eigen::MatrixXd>> global_cache;
  std::vector<std::shared_ptr<Layer>> layers;
  int num_layer;
  int input_size;
  int hidden_size;
  int num_classes;
  int reg_strength;
  std::unique_ptr<Layer> activation_function = std::make_unique<Relu>();

  void init_weights();
  void init_gradients();
  void init_regs();

public:
  std::unordered_map<std::string, Eigen::MatrixXd> gradients;
  std::unordered_map<std::string, Eigen::MatrixXd> params;
  std::unordered_map<std::string, double> regs;

  explicit Network(int num_layer = 2, int input_size = 28 * 28,
                   int hidden_size = 10, int num_classes = 10,
                   int reg_strength = 0);
  Eigen::MatrixXd forward(const Eigen::MatrixXd &input);
  std::unordered_map<std::string, Eigen::MatrixXd>
  backward(const Eigen::MatrixXd &dy);
  void add_layer(const std::shared_ptr<Layer> &layer) {
    layers.push_back(layer);
  }
  std::vector<std::shared_ptr<Layer>> &get_layers() { return layers; }

  std::unordered_map<std::string, Eigen::MatrixXd> get_params() {
    return params;
  }
  std::unordered_map<std::string, std::vector<Eigen::MatrixXd>> get_cache() {
    return global_cache;
  }
  // for testing purposes, set all cells to same value
  void set_params(double value);
};

#endif

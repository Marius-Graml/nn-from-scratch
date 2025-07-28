
#ifndef LAYER_H
#define LAYER_H

#include "../../../libs/Eigen/Dense"

class Layer {

private:
public:
  virtual std::vector<Eigen::MatrixXd> forward(Eigen::MatrixXd Y) = 0;

  virtual Eigen::MatrixXd backward(Eigen::MatrixXd dout,
                                   std::vector<Eigen::MatrixXd> cache) = 0;
};

#endif
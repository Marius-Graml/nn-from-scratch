#ifndef BATCH_H
#define BATCH_H

#include "../../libs/Eigen/Dense"

class Batch {
public:
  Eigen::MatrixXd data;
  Eigen::MatrixXd labels;

  // Constructor to initialize the attributes
  Batch(Eigen::MatrixXd data, Eigen::MatrixXd labels)
      : data(data), labels(labels) {}
};

#endif
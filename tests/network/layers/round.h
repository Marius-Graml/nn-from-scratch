#ifndef ROUND_H
#define ROUND_H

#include "../../../libs/Eigen/Dense"

inline Eigen::MatrixXd
roundMatrixToFiveDecimalPlaces(const Eigen::MatrixXd &matrix) {
  Eigen::MatrixXd rounded =
      matrix.unaryExpr([](double x) { return std::round(x * 1e5) / 1e5; });
  return rounded;
}

inline Eigen::VectorXd
roundVectorToFiveDecimalPlaces(const Eigen::VectorXd &vector) {
  Eigen::VectorXd rounded =
      vector.unaryExpr([](double x) { return std::round(x * 1e5) / 1e5; });
  return rounded;
}

#endif
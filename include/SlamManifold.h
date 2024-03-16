/**
 * CeresOptimizer
 *
 * Dknt 2024.3
 */

#ifndef SLAMMANIFOLD_H
#define SLAMMANIFOLD_H

#include <ceres/ceres.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>

/// @brief Right variation on manifold SO3
/// When using this manifold class, jacobian should be calculated for Lie
/// algebra in cost function, hence simply set jacobian as identity in
/// PlusJacobian
class QuaternionManifoldIdentity : public ceres::Manifold {
 public:
  virtual int AmbientSize() const override { return 4; }

  virtual int TangentSize() const override { return 3; }

  virtual bool Plus(const double* x, const double* delta,
                    double* x_plus_delta) const override {
    Eigen::Map<const Eigen::Quaterniond> ori(x);
    Eigen::Map<const Eigen::Vector3d> delta_v(delta);
    Eigen::Map<Eigen::Quaterniond> ori_new(x_plus_delta);
    // ori_new =
    //     Eigen::Quaterniond(ori.matrix() * Sophus::SO3d::exp(delta_v).matrix());

    Eigen::Quaterniond delta_q(1, delta_v[0] * 0.5, delta_v[1] * 0.5,
                               delta_v[2] * 0.5);
    ori_new = ori * delta_q;
    ori_new.normalize();
    return true;
  }

  virtual bool PlusJacobian(const double* x, double* jacobian) const override {
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> J(jacobian);
    J.setZero();
    J.block<3, 3>(0, 0).setIdentity();
    return true;
  }

  virtual bool Minus(const double* y, const double* x,
                     double* y_minus_x) const {
    std::cout << "Minus called. This shouldn't happen." << std::endl;
    return true;
  }

  virtual bool MinusJacobian(const double* x, double* jacobian) const override {
    std::cout << "MinusJacobian called. This shouldn't happen." << std::endl;
    return true;
  }
};

#endif

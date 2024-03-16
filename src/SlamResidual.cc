/**
 * Ceres Costfunctions
 *
 * Dknt 2024.3
 */

#include "SlamResidual.h"

// #include "SlamManifold.h"

/// TODO: optimize these codes for better runtime efficiency!

////////////////////////////////////////////////////////////////////////////////////
CostFunctionSE3ProjectionMonoPoseOnly::CostFunctionSE3ProjectionMonoPoseOnly(
    const Eigen::Vector3d pw, const double fx, const double fy, const double cx,
    const double cy, const Eigen::Vector2d measurement)
    : pw_(pw), fx_(fx), fy_(fy), cx_(cx), cy_(cy), measurement_(measurement) {
  K23_.setZero();
  K23_(0, 0) = fx_;
  K23_(0, 2) = cx_;
  K23_(1, 1) = fy_;
  K23_(1, 2) = cy_;
  information_matrix_sqrt_.setIdentity();
}

void CostFunctionSE3ProjectionMonoPoseOnly::SetInformationMatrix(
    const Eigen::Matrix2d& info) {
  information_matrix_sqrt_ = info;
}

bool CostFunctionSE3ProjectionMonoPoseOnly::Evaluate(
    double const* const* parameters, double* residuals,
    double** jacobians) const {
  Eigen::Map<const Eigen::Vector3d> tcw(parameters[0]);
  Eigen::Map<const Eigen::Quaterniond> qcw(parameters[1]);
  Eigen::Map<Eigen::Vector2d> error(residuals);

  Eigen::Vector3d Pc = qcw * pw_ + tcw;
  double z_inv = 1 / Pc[2];
  error = measurement_ - (K23_ * Pc * z_inv);
  error = information_matrix_sqrt_ * error;

  if (jacobians) {
    double z_inv2 = z_inv * z_inv;
    Eigen::Matrix<double, 2, 3> du_dPc;
    du_dPc.setZero();
    du_dPc(0, 0) = fx_ * z_inv;
    du_dPc(0, 2) = -fx_ * Pc[0] * z_inv2;
    du_dPc(1, 1) = fy_ * z_inv;
    du_dPc(1, 2) = -fy_ * Pc[1] * z_inv2;

    // tcw jacobian
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[0]);
      J.setZero();
      J = -du_dPc;
      J = information_matrix_sqrt_ * J;
    }
    // Rcw jocabian,  variation on SO(3)
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J(jacobians[1]);
      J.setZero();
      J.block<2, 3>(0, 0) = du_dPc * qcw.matrix() * Sophus::SO3d::hat(pw_);
      J = information_matrix_sqrt_ * J;
    }
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////
CostFunctionSE3ProjectionStereoPoseOnly::
    CostFunctionSE3ProjectionStereoPoseOnly(const Eigen::Vector3d pw,
                                            const double fx, const double fy,
                                            const double cx, const double cy,
                                            const double bf,
                                            const Eigen::Vector3d measurement)
    : pw_(pw),
      fx_(fx),
      fy_(fy),
      cx_(cx),
      cy_(cy),
      bf_(bf),
      measurement_(measurement) {
  K33_.setZero();
  K33_(0, 0) = fx_;
  K33_(0, 2) = cx_;
  K33_(1, 1) = fy_;
  K33_(1, 2) = cy_;
  K33_(2, 0) = fx_;
  // K33_(2, 2) = cx_;  // This should be refined when evaluate
  information_matrix_sqrt_.setIdentity();
}

void CostFunctionSE3ProjectionStereoPoseOnly::SetInformationMatrix(
    const Eigen::Matrix3d& info) {
  information_matrix_sqrt_ = info;
}

bool CostFunctionSE3ProjectionStereoPoseOnly::Evaluate(
    double const* const* parameters, double* residuals,
    double** jacobians) const {
  Eigen::Map<const Eigen::Vector3d> tcw(parameters[0]);
  Eigen::Map<const Eigen::Quaterniond> qcw(parameters[1]);
  Eigen::Map<Eigen::Vector3d> error(residuals);

  Eigen::Vector3d Pc = qcw * pw_ + tcw;
  double z_inv = 1 / Pc[2];
  K33_(2, 2) = cx_ - bf_ * z_inv;
  error = measurement_ - (K33_ * Pc * z_inv);
  error = information_matrix_sqrt_ * error;

  if (jacobians) {
    double z_inv2 = z_inv * z_inv;
    Eigen::Matrix<double, 3, 3> du_dPc;
    du_dPc.setZero();
    du_dPc(0, 0) = fx_ * z_inv;
    du_dPc(0, 2) = -fx_ * Pc[0] * z_inv2;
    du_dPc(1, 1) = fy_ * z_inv;
    du_dPc(1, 2) = -fy_ * Pc[1] * z_inv2;
    du_dPc(2, 0) = fx_ * z_inv;
    du_dPc(2, 2) = -fx_ * Pc[0] * z_inv2 + bf_ * z_inv2;

    // tcw jacobian
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
      J.setZero();
      J = -du_dPc;
      J = information_matrix_sqrt_ * J;
    }
    // Rcw jocabian, variation on SO(3)
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(jacobians[1]);
      J.setZero();
      J.block<3, 3>(0, 0) = du_dPc * qcw.matrix() * Sophus::SO3d::hat(pw_);
      J = information_matrix_sqrt_ * J;
    }
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////
CostFunctionSE3ProjectionMono::CostFunctionSE3ProjectionMono(
    const double fx, const double fy, const double cx, const double cy,
    const Eigen::Vector2d measurement)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), measurement_(measurement) {
  K23_.setZero();
  K23_(0, 0) = fx_;
  K23_(0, 2) = cx_;
  K23_(1, 1) = fy_;
  K23_(1, 2) = cy_;
  information_matrix_sqrt_.setIdentity();
}

void CostFunctionSE3ProjectionMono::SetInformationMatrix(
    const Eigen::Matrix2d& info) {
  information_matrix_sqrt_ = info;
}

bool CostFunctionSE3ProjectionMono::Evaluate(double const* const* parameters,
                                             double* residuals,
                                             double** jacobians) const {
  Eigen::Map<const Eigen::Vector3d> tcw(parameters[0]);
  Eigen::Map<const Eigen::Quaterniond> qcw(parameters[1]);
  Eigen::Map<const Eigen::Vector3d> pw(parameters[2]);
  Eigen::Map<Eigen::Vector2d> error(residuals);

  Eigen::Vector3d Pc = qcw * pw + tcw;
  double z_inv = 1 / Pc[2];
  error = measurement_ - (K23_ * Pc * z_inv);
  error = information_matrix_sqrt_ * error;

  if (jacobians) {
    double z_inv2 = z_inv * z_inv;
    Eigen::Matrix<double, 2, 3> du_dPc;
    du_dPc.setZero();
    du_dPc(0, 0) = fx_ * z_inv;
    du_dPc(0, 2) = -fx_ * Pc[0] * z_inv2;
    du_dPc(1, 1) = fy_ * z_inv;
    du_dPc(1, 2) = -fy_ * Pc[1] * z_inv2;

    // tcw jacobian
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[0]);
      J.setZero();
      J = -du_dPc;
      J = information_matrix_sqrt_ * J;
    }
    // Rcw jocabian,  variation on SO(3)
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J(jacobians[1]);
      J.setZero();
      J.block<2, 3>(0, 0) = du_dPc * qcw.matrix() * Sophus::SO3d::hat(pw);
      J = information_matrix_sqrt_ * J;
    }
    // pw jacobian
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[2]);
      J.setZero();
      J = -du_dPc * qcw.matrix();
      J = information_matrix_sqrt_ * J;
    }
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////
CostFunctionSE3ProjectionStereo::CostFunctionSE3ProjectionStereo(
    const double fx, const double fy, const double cx, const double cy,
    const double bf, const Eigen::Vector3d measurement)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), bf_(bf), measurement_(measurement) {
  K33_.setZero();
  K33_(0, 0) = fx_;
  K33_(0, 2) = cx_;
  K33_(1, 1) = fy_;
  K33_(1, 2) = cy_;
  K33_(2, 0) = fx_;
  // K33_(2, 2) = cx_;  // This should be refined when evaluate
  information_matrix_sqrt_.setIdentity();
}

void CostFunctionSE3ProjectionStereo::SetInformationMatrix(
    const Eigen::Matrix3d& info) {
  information_matrix_sqrt_ = info;
}

bool CostFunctionSE3ProjectionStereo::Evaluate(double const* const* parameters,
                                               double* residuals,
                                               double** jacobians) const {
  Eigen::Map<const Eigen::Vector3d> tcw(parameters[0]);
  Eigen::Map<const Eigen::Quaterniond> qcw(parameters[1]);
  Eigen::Map<const Eigen::Vector3d> pw(parameters[2]);
  Eigen::Map<Eigen::Vector3d> error(residuals);

  Eigen::Vector3d Pc = qcw * pw + tcw;
  double z_inv = 1 / Pc[2];
  K33_(2, 2) = cx_ - bf_ * z_inv;
  error = measurement_ - (K33_ * Pc * z_inv);
  error = information_matrix_sqrt_ * error;

  if (jacobians) {
    double z_inv2 = z_inv * z_inv;
    Eigen::Matrix<double, 3, 3> du_dPc;
    du_dPc.setZero();
    du_dPc(0, 0) = fx_ * z_inv;
    du_dPc(0, 2) = -fx_ * Pc[0] * z_inv2;
    du_dPc(1, 1) = fy_ * z_inv;
    du_dPc(1, 2) = -fy_ * Pc[1] * z_inv2;
    du_dPc(2, 0) = fx_ * z_inv;
    du_dPc(2, 2) = -fx_ * Pc[0] * z_inv2 + bf_ * z_inv2;

    // tcw jacobian
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
      J.setZero();
      J = -du_dPc;
      J = information_matrix_sqrt_ * J;
    }
    // Rcw jocabian, variation on SO(3)
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(jacobians[1]);
      J.setZero();
      J.block<3, 3>(0, 0) = du_dPc * qcw.matrix() * Sophus::SO3d::hat(pw);
      J = information_matrix_sqrt_ * J;
    }
    // pw jacobian
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[2]);
      J.setZero();
      J = -du_dPc * qcw.matrix();
      J = information_matrix_sqrt_ * J;
    }
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////////
CostFunctionPoseGraphSim3::CostFunctionPoseGraphSim3(
    const Eigen::Vector3d t12, const Eigen::Quaterniond q12, const double s12)
    : t12_(t12), q12_(q12), s12_(s12) {}

void CostFunctionPoseGraphSim3::SetInformationMatrix(
    const Eigen::Matrix<double, 7, 7>& info) {
  information_matrix_sqrt_ = info;
}

bool CostFunctionPoseGraphSim3::Evaluate(double const* const* parameters,
                                         double* residuals,
                                         double** jacobians) const {
  Eigen::Map<const Eigen::Vector3d> t1w(parameters[0]);
  Eigen::Map<const Eigen::Quaterniond> q1w(parameters[1]);
  const double& s1w = parameters[2][0];
  Eigen::Map<const Eigen::Vector3d> t2w(parameters[3]);
  Eigen::Map<const Eigen::Quaterniond> q2w(parameters[4]);
  const double& s2w = parameters[5][0];

  Eigen::Map<Eigen::Vector<double, 7>> error(residuals);
  Eigen::Map<Eigen::Vector3d> e_t(residuals);
  Eigen::Map<Eigen::Vector3d> e_r(residuals + 3);
  double& e_s = residuals[6];

  e_t = -s1w / s2w * q1w.matrix() * q2w.inverse().matrix() * t2w + t1w - t12_;
  Eigen::Vector3d err_ori =
      Sophus::SO3d((q12_.inverse() * q1w * q2w.inverse()).normalized().matrix())
          .log();
  e_r = err_ori;
  e_s = std::log(s1w / (s12_ * s2w));
  error = information_matrix_sqrt_ * error;
  if (jacobians) {
    // de / dt1w
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 7, 3, Eigen::RowMajor>> J(jacobians[0]);
      J.setZero();
      // det / dt1w
      J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    }
    // de / dq1w
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 7, 4, Eigen::RowMajor>> J(jacobians[1]);
      J.setZero();
      // det / dq1w
      J.block<3, 3>(0, 0) = q1w.matrix() *
                            Sophus::SO3d::hat(q2w.inverse().matrix() * t2w) *
                            s1w / s2w;
      // der / dq1w
      J.block<3, 3>(3, 0) =
          (Eigen::Matrix3d::Identity() + 0.5 * Sophus::SO3d::hat(err_ori)) *
          q2w.matrix();
    }
    // de / ds1w
    if (jacobians[2]) {
      Eigen::Map<Eigen::Vector<double, 7>> J(jacobians[2]);
      J.setZero();
      // det / de1w
      J.block<3, 1>(0, 0) = -q1w.matrix() * q2w.inverse().matrix() * t2w / s2w;
      // des / ds1w
      J(6, 0) = 1 / s1w;
    }
    // de / dt2w
    if (jacobians[3]) {
      Eigen::Map<Eigen::Matrix<double, 7, 3, Eigen::RowMajor>> J(jacobians[3]);
      J.setZero();
      // der / dt2w
      J.block<3, 3>(0, 0) = -q1w.matrix() * q2w.inverse().matrix() * s1w / s2w;
    }
    // de / dq2w
    if (jacobians[4]) {
      Eigen::Map<Eigen::Matrix<double, 7, 4, Eigen::RowMajor>> J(jacobians[4]);
      J.setZero();
      // det / dq2w
      J.block<3, 3>(0, 0) = -q1w.matrix() *
                            Sophus::SO3d::hat(q2w.inverse().matrix() * t2w) *
                            s1w / s2w;
      // der / dq2w
      J.block<3, 3>(3, 0) =
          -(Eigen::Matrix3d::Identity() + 0.5 * Sophus::SO3d::hat(err_ori)) *
          q2w.matrix();
    }
    // de / ds2w
    if (jacobians[5]) {
      Eigen::Map<Eigen::Vector<double, 7>> J(jacobians[5]);
      J.setZero();
      // det / ds2w
      J.block<3, 1>(0, 0) =
          q1w.matrix() * q2w.inverse().matrix() * t2w * s1w / (s2w * s2w);
      // des/ ds2w
      J(6, 0) = -1 / s2w;
    }
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////////
CostFunctionPoseGraphSE3::CostFunctionPoseGraphSE3(const Eigen::Vector3d t12,
                                                   const Eigen::Quaterniond q12)
    : t12_(t12), q12_(q12) {}

void CostFunctionPoseGraphSE3::SetInformationMatrix(
    const Eigen::Matrix<double, 6, 6>& info) {
  information_matrix_sqrt_ = info;
}

bool CostFunctionPoseGraphSE3::Evaluate(double const* const* parameters,
                                        double* residuals,
                                        double** jacobians) const {
  Eigen::Map<const Eigen::Vector3d> t1w(parameters[0]);
  Eigen::Map<const Eigen::Quaterniond> q1w(parameters[1]);
  Eigen::Map<const Eigen::Vector3d> t2w(parameters[2]);
  Eigen::Map<const Eigen::Quaterniond> q2w(parameters[3]);

  Eigen::Map<Eigen::Vector<double, 6>> error(residuals);
  Eigen::Map<Eigen::Vector3d> e_t(residuals);
  Eigen::Map<Eigen::Vector3d> e_r(residuals + 3);

  e_t = q1w.matrix() * q2w.inverse().matrix() * t2w + t1w - t12_;
  Eigen::Vector3d err_ori =
      Sophus::SO3d((q12_.inverse() * q1w * q2w.inverse()).normalized().matrix())
          .log();
  e_r = err_ori;
  error = information_matrix_sqrt_ * error;
  if (jacobians) {
    // de / dt1w
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[0]);
      J.setZero();
      // det / dt1w
      J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    }
    // de / dq1w
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> J(jacobians[1]);
      J.setZero();
      // det / dq1w
      J.block<3, 3>(0, 0) =
          q1w.matrix() * Sophus::SO3d::hat(q2w.inverse().matrix() * t2w);
      // der / dq1w
      J.block<3, 3>(3, 0) =
          (Eigen::Matrix3d::Identity() + 0.5 * Sophus::SO3d::hat(err_ori)) *
          q2w.matrix();
    }
    // de / dt2w
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[2]);
      J.setZero();
      // der / dt2w
      J.block<3, 3>(0, 0) = -q1w.matrix() * q2w.inverse().matrix();
    }
    // de / dq2w
    if (jacobians[3]) {
      Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> J(jacobians[3]);
      J.setZero();
      // det / dq2w
      J.block<3, 3>(0, 0) =
          -q1w.matrix() * Sophus::SO3d::hat(q2w.inverse().matrix() * t2w);
      // der / dq2w
      J.block<3, 3>(3, 0) =
          -(Eigen::Matrix3d::Identity() + 0.5 * Sophus::SO3d::hat(err_ori)) *
          q2w.matrix();
    }
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////////
CostFunctionSim3ProjectionMono::CostFunctionSim3ProjectionMono(
    const Eigen::Vector3d p2, const double fx, const double fy, const double cx,
    const double cy, const Eigen::Vector2d measurement)
    : p2_(p2), fx_(fx), fy_(fy), cx_(cx), cy_(cy), measurement_(measurement) {
  K23_.setZero();
  K23_(0, 0) = fx_;
  K23_(0, 2) = cx_;
  K23_(1, 1) = fy_;
  K23_(1, 2) = cy_;
  information_matrix_sqrt_.setIdentity();
}

void CostFunctionSim3ProjectionMono::SetInformationMatrix(
    const Eigen::Matrix2d& info) {
  information_matrix_sqrt_ = info;
}

bool CostFunctionSim3ProjectionMono::Evaluate(double const* const* parameters,
                                              double* residuals,
                                              double** jacobians) const {
  Eigen::Map<const Eigen::Vector3d> t12(parameters[0]);
  Eigen::Map<const Eigen::Quaterniond> q12(parameters[1]);
  const double& s12 = parameters[2][0];
  Eigen::Map<Eigen::Vector2d> error(residuals);

  Eigen::Vector3d p1 = q12 * p2_ * s12 + t12;
  double z_inv = 1 / p1[2];
  error = measurement_ - (K23_ * p1 * z_inv);
  error = information_matrix_sqrt_ * error;

  if (jacobians) {
    double z_inv2 = z_inv * z_inv;
    Eigen::Matrix<double, 2, 3> du_dP1;
    du_dP1.setZero();
    du_dP1(0, 0) = fx_ * z_inv;
    du_dP1(0, 2) = -fx_ * p1[0] * z_inv2;
    du_dP1(1, 1) = fy_ * z_inv;
    du_dP1(1, 2) = -fy_ * p1[1] * z_inv2;

    // de / dt
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[0]);
      J.setZero();
      J = -du_dP1;
      J = information_matrix_sqrt_ * J;
    }
    // de / dr
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J(jacobians[1]);
      J.setZero();
      J.block<2, 3>(0, 0) = du_dP1 * Sophus::SO3d::hat(p2_) * s12;
      J = information_matrix_sqrt_ * J;
    }
    // de / ds
    if (jacobians[2]) {
      Eigen::Map<Eigen::Vector<double, 2>> J(jacobians[2]);
      J.setZero();
      J = -du_dP1 * q12.matrix() * p2_;
      J = information_matrix_sqrt_ * J;
    }
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////////
CostFunctionSim3InverseProjectionMono::CostFunctionSim3InverseProjectionMono(
    const Eigen::Vector3d p1, const double fx, const double fy, const double cx,
    const double cy, const Eigen::Vector2d measurement)
    : p1_(p1), fx_(fx), fy_(fy), cx_(cx), cy_(cy), measurement_(measurement) {
  K23_.setZero();
  K23_(0, 0) = fx_;
  K23_(0, 2) = cx_;
  K23_(1, 1) = fy_;
  K23_(1, 2) = cy_;
  information_matrix_sqrt_.setIdentity();
}

void CostFunctionSim3InverseProjectionMono::SetInformationMatrix(
    const Eigen::Matrix2d& info) {
  information_matrix_sqrt_ = info;
}

bool CostFunctionSim3InverseProjectionMono::Evaluate(
    double const* const* parameters, double* residuals,
    double** jacobians) const {
  Eigen::Map<const Eigen::Vector3d> t12(parameters[0]);
  Eigen::Map<const Eigen::Quaterniond> q12(parameters[1]);
  const double& s12 = parameters[2][0];
  Eigen::Map<Eigen::Vector2d> error(residuals);
  Eigen::Quaterniond q21 = q12.inverse();

  Eigen::Vector3d p2 = q21 * (p1_ - t12) / s12;
  double z_inv = 1 / p2[2];
  error = measurement_ - (K23_ * p2 * z_inv);
  error = information_matrix_sqrt_ * error;

  if (jacobians) {
    double z_inv2 = z_inv * z_inv;
    Eigen::Matrix<double, 2, 3> du_dP2;
    du_dP2.setZero();
    du_dP2(0, 0) = fx_ * z_inv;
    du_dP2(0, 2) = -fx_ * p2[0] * z_inv2;
    du_dP2(1, 1) = fy_ * z_inv;
    du_dP2(1, 2) = -fy_ * p2[1] * z_inv2;

    // de / dt
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[0]);
      J.setZero();
      J = du_dP2 * q21.matrix() / s12;
      J = information_matrix_sqrt_ * J;
    }
    // de / dr
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J(jacobians[1]);
      J.setZero();
      J.block<2, 3>(0, 0) = du_dP2 * Sophus::SO3d::hat(q21.matrix() * (t12 - p1_)) / s12;
      J = information_matrix_sqrt_ * J;
    }
    // de / ds
    if (jacobians[2]) {
      Eigen::Map<Eigen::Vector<double, 2>> J(jacobians[2]);
      J.setZero();
      J = du_dP2 * q21.matrix() * (p1_ - t12) / (s12 * s12);
      J = information_matrix_sqrt_ * J;
    }
  }
  return true;
}

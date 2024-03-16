/**
 * CeresOptimizer
 *
 * Dknt 2024.3
 */

#ifndef SLAMRESIDUAL_H
#define SLAMRESIDUAL_H

#include <ceres/ceres.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>

class CostFunctionSE3ProjectionMonoPoseOnly
    : public ceres::SizedCostFunction<2, 3, 4> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  CostFunctionSE3ProjectionMonoPoseOnly(const Eigen::Vector3d pw,
                                        const double fx, const double fy,
                                        const double cx, const double cy,
                                        const Eigen::Vector2d measurement);

  void SetInformationMatrix(const Eigen::Matrix2d& info);

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const override;

  Eigen::Vector3d pw_;  // MapPoint position in global coordinate system, which
                        // remains constant during optimization
  double fx_, fy_, cx_, cy_;
  Eigen::Vector2d measurement_;  // 2D projection on image  (u, v)
  Eigen::Matrix2d information_matrix_sqrt_;
  Eigen::Matrix<double, 2, 3> K23_;

 private:
};

class CostFunctionSE3ProjectionStereoPoseOnly
    : public ceres::SizedCostFunction<3, 3, 4> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  CostFunctionSE3ProjectionStereoPoseOnly(const Eigen::Vector3d pw,
                                          const double fx, const double fy,
                                          const double cx, const double cy,
                                          const double bf,
                                          const Eigen::Vector3d measurement);

  void SetInformationMatrix(const Eigen::Matrix3d& info);

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const override;

  Eigen::Vector3d pw_;  // MapPoint position in global coordinate system, which
                        // remains constant during optimization
  double fx_, fy_, cx_, cy_, bf_;
  Eigen::Vector3d
      measurement_;  // 3D projection on stereo image  (u, v, u_right)
  Eigen::Matrix3d information_matrix_sqrt_;
  mutable Eigen::Matrix<double, 3, 3> K33_;

 private:
};

class CostFunctionSE3ProjectionMono
    : public ceres::SizedCostFunction<2, 3, 4, 3> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  CostFunctionSE3ProjectionMono(const double fx, const double fy,
                                const double cx, const double cy,
                                const Eigen::Vector2d measurement);

  void SetInformationMatrix(const Eigen::Matrix2d& info);

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const override;

  inline void SetKeyFrameAndMapPoint(void* kf, void* mp) {
    kf_ptr_ = kf;
    mp_ptr_ = mp;
  }

  double fx_, fy_, cx_, cy_;
  Eigen::Vector2d measurement_;  // 2D projection on image  (u, v)
  Eigen::Matrix2d information_matrix_sqrt_;
  Eigen::Matrix<double, 2, 3> K23_;
  void* kf_ptr_;
  void* mp_ptr_;

 private:
};

class CostFunctionSE3ProjectionStereo
    : public ceres::SizedCostFunction<3, 3, 4, 3> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  CostFunctionSE3ProjectionStereo(const double fx, const double fy,
                                  const double cx, const double cy,
                                  const double bf,
                                  const Eigen::Vector3d measurement);

  void SetInformationMatrix(const Eigen::Matrix3d& info);

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const override;

  inline void SetKeyFrameAndMapPoint(void* kf, void* mp) {
    kf_ptr_ = kf;
    mp_ptr_ = mp;
  }

  double fx_, fy_, cx_, cy_, bf_;
  Eigen::Vector3d
      measurement_;  // 3D projection on stereo image  (u, v, u_right)
  Eigen::Matrix3d information_matrix_sqrt_;
  mutable Eigen::Matrix<double, 3, 3> K33_;
  void* kf_ptr_;
  void* mp_ptr_;

 private:
};

class CostFunctionPoseGraphSim3
    : public ceres::SizedCostFunction<7, 3, 4, 1, 3, 4, 1> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  CostFunctionPoseGraphSim3(const Eigen::Vector3d t12,
                            const Eigen::Quaterniond q12, const double s12);

  void SetInformationMatrix(const Eigen::Matrix<double, 7, 7>& info);

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const override;

  Eigen::Vector3d t12_;
  Eigen::Quaterniond q12_;
  double s12_;
  Eigen::Matrix<double, 7, 7> information_matrix_sqrt_ =
      Eigen::Matrix<double, 7, 7>::Identity();

 private:
};

class CostFunctionPoseGraphSE3
    : public ceres::SizedCostFunction<6, 3, 4, 3, 4> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  CostFunctionPoseGraphSE3(const Eigen::Vector3d t12,
                           const Eigen::Quaterniond q12);
  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const override;
  void SetInformationMatrix(const Eigen::Matrix<double, 6, 6>& info);

  Eigen::Vector3d t12_;
  Eigen::Quaterniond q12_;
  double s12_;
  Eigen::Matrix<double, 6, 6> information_matrix_sqrt_ =
      Eigen::Matrix<double, 6, 6>::Identity();

 private:
};

class CostFunctionSim3ProjectionMono
    : public ceres::SizedCostFunction<2, 3, 4, 1> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  CostFunctionSim3ProjectionMono(const Eigen::Vector3d p2, const double fx,
                                 const double fy, const double cx,
                                 const double cy,
                                 const Eigen::Vector2d measurement);

  void SetInformationMatrix(const Eigen::Matrix2d& info);

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const override;

  Eigen::Vector3d p2_;
  double fx_, fy_, cx_, cy_;
  Eigen::Vector2d measurement_;  // 2D projection on image  (u, v)
  Eigen::Matrix2d information_matrix_sqrt_;
  Eigen::Matrix<double, 2, 3> K23_;

 private:
};

class CostFunctionSim3InverseProjectionMono
    : public ceres::SizedCostFunction<2, 3, 4, 1> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  CostFunctionSim3InverseProjectionMono(const Eigen::Vector3d p1,
                                        const double fx, const double fy,
                                        const double cx, const double cy,
                                        const Eigen::Vector2d measurement);

  void SetInformationMatrix(const Eigen::Matrix2d& info);

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const override;

  Eigen::Vector3d p1_;
  double fx_, fy_, cx_, cy_;
  Eigen::Vector2d measurement_;  // 2D projection on image  (u, v)
  Eigen::Matrix2d information_matrix_sqrt_;
  Eigen::Matrix<double, 2, 3> K23_;

 private:
};

#endif

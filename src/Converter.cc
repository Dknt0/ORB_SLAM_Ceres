/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University
 * of Zaragoza) For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Converter.h"

namespace ORB_SLAM2 {

/// @brief OpenCV 描述子矩阵转描述子向量
/// @param Descriptors
/// @return
std::vector<cv::Mat> Converter::toDescriptorVector(const cv::Mat &Descriptors) {
  std::vector<cv::Mat> vDesc;
  vDesc.reserve(Descriptors.rows);
  for (int j = 0; j < Descriptors.rows; j++)
    vDesc.push_back(Descriptors.row(j));

  return vDesc;
}

// /// @brief cv::Mat SE(3) 转 g2o::SE3Quat
// /// @param cvT
// /// @return
// g2o::SE3Quat Converter::toSE3Quat(const cv::Mat &cvT) {
//   Eigen::Matrix<double, 3, 3> R;
//   R << cvT.at<float>(0, 0), cvT.at<float>(0, 1), cvT.at<float>(0, 2),
//       cvT.at<float>(1, 0), cvT.at<float>(1, 1), cvT.at<float>(1, 2),
//       cvT.at<float>(2, 0), cvT.at<float>(2, 1), cvT.at<float>(2, 2);

//   Eigen::Matrix<double, 3, 1> t(cvT.at<float>(0, 3), cvT.at<float>(1, 3),
//                                 cvT.at<float>(2, 3));

//   return g2o::SE3Quat(R, t);
// }

// /// @brief g2o::SE3Quat 转 cv::Mat SE(3)
// /// @param SE3
// /// @return
// cv::Mat Converter::toCvMat(const g2o::SE3Quat &SE3) {
//   Eigen::Matrix<double, 4, 4> eigMat = SE3.to_homogeneous_matrix();
//   return toCvMat(eigMat);
// }

/// @brief Sim3 转 cv::Mat Sim(3)
/// @param Sim3
/// @return
cv::Mat Converter::toCvMat(const Sim3 &Sim3) {
  Eigen::Matrix3d eigR = Sim3.rotation().toRotationMatrix();
  Eigen::Vector3d eigt = Sim3.translation();
  double s = Sim3.scale();
  return toCvSE3(s * eigR, eigt);
}

/// @brief Eigen::Matrix4d 转 cv::Mat 4*4
/// @param m
/// @return
cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 4, 4> &m) {
  cv::Mat cvMat(4, 4, CV_32F);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) cvMat.at<float>(i, j) = m(i, j);

  return cvMat.clone();
}

/// @brief Eigen::Matrix3d 转 cv::Mat 3*3
/// @param m
/// @return
cv::Mat Converter::toCvMat(const Eigen::Matrix3d &m) {
  cv::Mat cvMat(3, 3, CV_32F);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) cvMat.at<float>(i, j) = m(i, j);

  return cvMat.clone();
}

/// @brief Eigen::Vector3d 转 cv::Mat 3*1
/// @param m
/// @return
cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 3, 1> &m) {
  cv::Mat cvMat(3, 1, CV_32F);
  for (int i = 0; i < 3; i++) cvMat.at<float>(i) = m(i);

  return cvMat.clone();
}

/// @brief Eigen::Matrix3d 旋转 Eigen::Vector3d 平移转 cv::Mat SE(3)
/// @param R
/// @param t
/// @return
cv::Mat Converter::toCvSE3(const Eigen::Matrix<double, 3, 3> &R,
                           const Eigen::Matrix<double, 3, 1> &t) {
  cv::Mat cvMat = cv::Mat::eye(4, 4, CV_32F);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      cvMat.at<float>(i, j) = R(i, j);
    }
  }
  for (int i = 0; i < 3; i++) {
    cvMat.at<float>(i, 3) = t(i);
  }

  return cvMat.clone();
}

/// @brief cv::Mat 3*1 转 Eigen::Vector3d
/// @param cvVector
/// @return
Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Mat &cvVector) {
  Eigen::Matrix<double, 3, 1> v;
  v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

  return v;
}

/// @brief cv::Point3f 转 Eigen::Vector3d
/// @param cvPoint
/// @return
Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Point3f &cvPoint) {
  Eigen::Matrix<double, 3, 1> v;
  v << cvPoint.x, cvPoint.y, cvPoint.z;

  return v;
}

/// @brief cv::Mat SO(3) 转 Eigen::Matrix3d
/// @param cvMat3
/// @return
Eigen::Matrix<double, 3, 3> Converter::toMatrix3d(const cv::Mat &cvMat3) {
  Eigen::Matrix<double, 3, 3> M;

  M << cvMat3.at<float>(0, 0), cvMat3.at<float>(0, 1), cvMat3.at<float>(0, 2),
      cvMat3.at<float>(1, 0), cvMat3.at<float>(1, 1), cvMat3.at<float>(1, 2),
      cvMat3.at<float>(2, 0), cvMat3.at<float>(2, 1), cvMat3.at<float>(2, 2);

  return M;
}

/// @brief cv::Mat SO(3) 转 std::vector<double> 四元数
/// @param M
/// @return
std::vector<float> Converter::toQuaternion(const cv::Mat &M) {
  Eigen::Matrix<double, 3, 3> eigMat = toMatrix3d(M);
  Eigen::Quaterniond q(eigMat);

  std::vector<float> v(4);
  v[0] = q.x();
  v[1] = q.y();
  v[2] = q.z();
  v[3] = q.w();

  return v;
}

}  // namespace ORB_SLAM2

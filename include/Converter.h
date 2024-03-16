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

// ORB-SLAM2 中的矩阵、向量基本都是 cv::Mat 写的，而 g2o 基于 Eigen
// 所以需要类型转换
// ORB-SLAM3 中开始大量使用 Eigen

#ifndef CONVERTER_H
#define CONVERTER_H

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "sim3.h"



namespace ORB_SLAM2 {

/// @brief 一些类型转换函数
class Converter {
 public:
  // 为何都是 static ?

  static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

  // static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
  // static g2o::SE3Quat toSE3Quat(const Sim3 &gSim3);

  // static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
  static cv::Mat toCvMat(const Sim3 &Sim3);
  static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4> &m);
  static cv::Mat toCvMat(const Eigen::Matrix3d &m);
  static cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1> &m);
  static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3> &R,
                         const Eigen::Matrix<double, 3, 1> &t);

  static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Mat &cvVector);
  static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Point3f &cvPoint);
  static Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat &cvMat3);

  static std::vector<float> toQuaternion(const cv::Mat &M);
};

}  // namespace ORB_SLAM2

#endif  // CONVERTER_H

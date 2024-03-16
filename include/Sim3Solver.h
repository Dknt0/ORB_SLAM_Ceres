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

#ifndef SIM3SOLVER_H
#define SIM3SOLVER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "KeyFrame.h"

namespace ORB_SLAM2 {

// 应该是用于回环检测的
// 已知两关键帧，与共视地图点集合（可能存在外点），求帧间估计，作为初值使用

/// @brief Sim3 求解器
class Sim3Solver {
 public:
  Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2,
             const std::vector<MapPoint *> &vpMatched12,
             const bool bFixScale = true);

  void SetRansacParameters(double probability = 0.99, int minInliers = 6,
                           int maxIterations = 300);

  cv::Mat find(std::vector<bool> &vbInliers12, int &nInliers);
  cv::Mat iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers,
                  int &nInliers);

  cv::Mat GetEstimatedRotation();
  cv::Mat GetEstimatedTranslation();
  float GetEstimatedScale();

 protected:
  void ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C);

  void ComputeSim3(cv::Mat &P1, cv::Mat &P2);

  void CheckInliers();

  void Project(const std::vector<cv::Mat> &vP3Dw, std::vector<cv::Mat> &vP2D,
               cv::Mat Tcw, cv::Mat K);
  void FromCameraToImage(const std::vector<cv::Mat> &vP3Dc,
                         std::vector<cv::Mat> &vP2D, cv::Mat K);

 protected:
  /* KF, MP 输入 */
  // KeyFrames and matches
  KeyFrame *mpKF1;  // KF1
  KeyFrame *mpKF2;  // KF2
  std::vector<MapPoint *>
      mvpMatches12;  // KF1 MP 匹配到 KF2 MP  按照 KF1 KP 索引  原始输入
  int mN1;           // 输入 MP 匹配总数
  // Scale is fixed to 1 in the stereo/RGBD case
  bool mbFixScale;  // 固定尺度标志位  双目/深度为1
  /* KF, MP 有效匹配 */
  // 有效匹配指存在且有意义的 KF1 MP 到 KF2 MP 匹配，两边都不包括空指针与坏点
  int N;                         // 有效 MP 匹配总数
  std::vector<cv::Mat> mvX3Dc1;  // 有效匹配 KF1 MP KF1 相机系位置
  std::vector<cv::Mat> mvX3Dc2;  // 有效匹配 KF2 MP KF2 相机系位置
  std::vector<MapPoint *> mvpMapPoints1;  // 有效匹配 KF1 地图点
  std::vector<MapPoint *> mvpMapPoints2;  // 有效匹配 KF2 地图点
  std::vector<size_t> mvnIndices1;  // 有效匹配 匹配对在 MP 匹配集中的序号
  std::vector<size_t> mvSigmaSquare1;  // 有效匹配  KF1 KP 尺度平方  没有用到
  std::vector<size_t> mvSigmaSquare2;  // 有效匹配  KF2 KP 尺度平方  没有用到
  std::vector<size_t> mvnMaxError1;  // 有效匹配  KF1 MP 误差阈值 按照尺度确定
  std::vector<size_t> mvnMaxError2;  // 有效匹配  KF2 MP 误差阈值 按照尺度确定
  // Projections
  std::vector<cv::Mat> mvP1im1;  // 有效匹配  KF1 MP 投影到 KF1 像素坐标
  std::vector<cv::Mat> mvP2im2;  // 有效匹配  KF2 MP 投影到 KF2 像素坐标

  /* 当前估计 */
  // Current Estimation
  cv::Mat mR12i;                  // 当前估计 KF2 相对 KF1 姿态
  cv::Mat mt12i;                  // 当前估计 KF2 相对 KF1 位置
  float ms12i;                    // 当前估计 KF2 相对 KF1 尺度
  cv::Mat mT12i;                  // 当前估计 KF2 相对 KF1 位姿  Sim3
  cv::Mat mT21i;                  // 当前估计 KF1 相对 KF2 位姿  Sim3
  std::vector<bool> mvbInliersi;  // 当前估计 内点标志位
  int mnInliersi;                 // 当前估计 内点总数

  /* 当前 RANSAC 状态 */
  // Current Ransac State
  int mnIterations;                  // 总迭代次数
  std::vector<bool> mvbBestInliers;  // 最佳采样 内点标志位
  int mnBestInliers;                 // 最佳采样 内点总数
  cv::Mat mBestT12;                  // 最佳采样 KF2 相对 KF1 位姿
  cv::Mat mBestRotation;             // 最佳采样 姿态
  cv::Mat mBestTranslation;          // 最佳采样 位置
  float mBestScale;                  // 最佳采样 尺度

  /* RANSAC 参数 */
  // RANSAC probability
  double mRansacProb;  // =0.99 RANSAC 概率  用于确定最大迭代次数
  // RANSAC min inliers
  int mRansacMinInliers;  // =6 RANSAC 最小内点数
  // RANSAC max iterations
  int mRansacMaxIts;  // RANSAC 最大迭代次数
  // Indices for random selection
  std::vector<size_t> mvAllIndices;  // RANSAC 有效地图点序号  [0]=0, [1]=1 ...

  /* 误差阈值 */
  // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*mSigma2
  float mTh;      // 误差阈值
  float mSigma2;  // 尺度比例平方

  /* 相机内参 */
  // Calibration
  cv::Mat mK1;  // KF1 内参
  cv::Mat mK2;  // KF2 内参
};

}  // namespace ORB_SLAM2

#endif  // SIM3SOLVER_H
